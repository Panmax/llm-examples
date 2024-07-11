package me.jiapan.examples.llm.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.cohere.CohereScoringModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import me.jiapan.examples.llm.shared.Assistant;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static me.jiapan.examples.llm.shared.Utils.OPENAI_API_KEY;
import static me.jiapan.examples.llm.shared.Utils.startConversationWith;
import static me.jiapan.examples.llm.shared.Utils.toPath;

public class Example4_Advanced_RAG_with_ReRanking {

    /**
     * 本示例介绍了使用称为“重排序”的技术实现更高级的 RAG 应用程序
     * 通常，由 ContentRetriever 检索到的所有结果并不都与用户查询真正相关。
     * 这是因为，在初始检索阶段，通常更倾向于使用速度更快且成本效益更高的模型，特别是在处理大量数据时。
     * 这导致检索质量可能较低。向 LLM 提供无关信息可能代价高昂，并且在最坏情况下会导致幻觉。
     * 因此，在第二阶段，我们可以对第一阶段获得的结果进行重新排序，并使用更先进的模型（例如 Cohere Rerank）消除无关结果。
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant("documents/miles-of-smiles-terms-of-use.txt");

        // say "Hi". 观察第一阶段检索到的所有段落是如何被过滤掉的。
        // Then, ask "Can I cancel my reservation?" 观察除一个段以外的所有段是如何被过滤掉的。
        startConversationWith(assistant);
    }

    private static Assistant createAssistant(String documentPath) {

        Document document = loadDocument(toPath(documentPath), new TextDocumentParser());

        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        ingestor.ingest(document);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5) // NOTE: let's get more results
                .build();

        // https://docs.cohere.com/docs/rerank-2
        ScoringModel scoringModel = CohereScoringModel.builder()
                .apiKey(System.getenv("COHERE_API_KEY"))
                .logRequests(true)
                .logResponses(true)
                .build();

        ContentAggregator contentAggregator = ReRankingContentAggregator.builder()
                .scoringModel(scoringModel)
                .minScore(0.8) // NOTE: 我们希望向LLM展示仅与用户查询真正相关的部分
                .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                .contentAggregator(contentAggregator)
                .build();

        return AiServices.builder(Assistant.class)
                .chatLanguageModel(OpenAiChatModel.withApiKey(OPENAI_API_KEY))
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }
}
