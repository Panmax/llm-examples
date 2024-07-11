package me.jiapan.examples.llm.rag;

import me.jiapan.examples.llm.shared.Assistant;
import me.jiapan.examples.llm.shared.Utils;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.util.Collection;
import java.util.List;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;

public class Example5_Advanced_RAG_Skip_Retrieval {

    /**
     * 此示例演示了如何有条件地跳过检索。
     * 有时，检索是没有必要的，例如，当用户只是说“你好”时。
     * 决策可以通过多种方式实现：
     * - 使用规则（例如，根据用户的权限、位置等）。
     * - 使用关键词（例如，如果查询包含特定词语）。
     * - 使用语义相似性（请参见此存储库中的 EmbeddingModelTextClassifierExample）。
     * - 使用 LLM 做出决策。
     * 在这个例子中，我们将使用 LLM 来决定是否对用户查询进行检索。
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant();

        // First, say "Hi"
        // Notice how this query is not routed to any retrievers.

        // Now, ask "Can I cancel my reservation?"
        // This query has been routed to our retriever.
        Utils.startConversationWith(assistant);
    }

    private static Assistant createAssistant() {

        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore =
                embed(Utils.toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .apiKey(Utils.OPENAI_API_KEY)
                .build();

        // Let's create a query router.
        QueryRouter queryRouter = new QueryRouter() {

            private final PromptTemplate PROMPT_TEMPLATE = PromptTemplate.from(
                    "Is the following query related to the business of the car rental company? " +
                            "Answer only 'yes', 'no' or 'maybe'. " +
                            "Query: {{it}}"
            );

            @Override
            public Collection<ContentRetriever> route(Query query) {

                Prompt prompt = PROMPT_TEMPLATE.apply(query.text());

                AiMessage aiMessage = chatLanguageModel.generate(prompt.toUserMessage()).content();
                System.out.println("LLM decided: " + aiMessage.text());

                if (aiMessage.text().toLowerCase().contains("no")) {
                    return emptyList();
                }

                return singletonList(contentRetriever);
            }
        };

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        return AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    private static EmbeddingStore<TextSegment> embed(Path documentPath, EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(documentPath, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }
}
