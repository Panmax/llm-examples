package me.jiapan.examples.llm.rag;

import me.jiapan.examples.llm.shared.Utils;
import me.jiapan.examples.llm.shared.Assistant;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

public class Example2_Advanced_RAG_with_Query_Compression {

    /**
     * 本示例说明了使用一种称为“查询压缩”的技术实现更复杂的 RAG 应用程序。
     * 通常，用户的查询是一个后续问题，它回顾了对话的早期部分，并且缺乏有效检索所需的所有细节。
     * 例如，考虑以下对话：
     * User: What is the legacy of John Doe?
     * AI: John Doe was a...
     * User: When was he born?
     * 在这种情况下，使用带有类似“他什么时候出生？”这样的查询进行基本RAG方法可能无法找到关于 John Doe 的文章，因为查询中不包含“John Doe”。
     * 查询压缩涉及将用户的查询和之前的对话结合起来，然后要求 LLM 将其“压缩”成一个单一、自包含的问题。
     * LLM 应该生成类似于“John Doe何时出生？”的问题。
     * 这种方法增加了一些延迟和成本，但显著提高了 RAG 流程的质量。
     * 用于压缩的 LLM 不必与用于对话的一样。例如，可以使用经过摘要训练的小型本地模型。
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant("documents/biography-of-john-doe.txt");

        // What is the legacy of John Doe?
        // When was he born?
        // 查看日志:
        // 第一个查询没有被压缩，因为没有前面的上下文可以压缩。
        // 第二个查询被压缩成类似 John Doe birthdate
        Utils.startConversationWith(assistant);
    }

    private static Assistant createAssistant(String documentPath) {

        Document document = FileSystemDocumentLoader.loadDocument(Utils.toPath(documentPath), new TextDocumentParser());

        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        ingestor.ingest(document);

        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .apiKey(Utils.OPENAI_API_KEY)
                .build();

        // 我们将创建一个CompressingQueryTransformer，它负责压缩用户的查询和之前的对话为一个独立的查询。
        // 这应该会显著提高检索过程的质量。
        // NOTE: 实现方式是通过调用大模型，让大模型做总结，可以点进去看一下prompt
        QueryTransformer queryTransformer = new CompressingQueryTransformer(chatLanguageModel);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // RetrievalAugmentor 是 LangChain4j 中 RAG 流程的入口
        // 它可以配置以根据您的需求自定义 RAG 行为
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(queryTransformer)
                .contentRetriever(contentRetriever)
                .build();

        return AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }
}
