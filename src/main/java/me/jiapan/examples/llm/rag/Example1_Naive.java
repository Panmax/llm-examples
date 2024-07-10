package me.jiapan.examples.llm.rag;

import me.jiapan.examples.llm.shared.Assistant;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

import static me.jiapan.examples.llm.shared.Utils.OPENAI_API_KEY;
import static me.jiapan.examples.llm.shared.Utils.startConversationWith;
import static me.jiapan.examples.llm.shared.Utils.toPath;
import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

public class Example1_Naive {
    public static void main(String[] args) {

        Assistant assistant = createAssistant("documents/miles-of-smiles-terms-of-use.txt");

        // - Can I cancel my reservation?
        // - I had an accident, should I pay extra?
        startConversationWith(assistant);
    }

    private static Assistant createAssistant(String documentPath) {

        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-3.5-turbo")
                .build();


        // 加载一个我们想要用于RAG的文档
        // 我们将使用一家虚构的汽车租赁公司“Miles of Smiles”的使用条款。
        // 在这个例子中，我们只导入一个文档，但你可以根据需要加载多个文档。
        // 此外，支持解析多种类型的文档：文本, pdf, doc, xls, ppt。
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(toPath(documentPath), documentParser);


        // 将这个文档分成更小的部分，也称为“块” chunks
        // 这种方法使我们能够在响应用户查询时仅发送相关的段落给LLM，
        // 而不是整个文档。例如，如果用户询问取消政策，我们将识别并仅发送与取消相关的段落。
        // 一个好的启动方式是使用递归文档拆分器，最初尝试按段落拆分。
        // 如果一个段落太大而无法放入单个部分，拆分器将递归地按换行符、然后按句子、最后按单词进行划分（如果有必要），
        // 以确保每一片文本都能适应到一个单独的部分中。
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        // 现在，我们需要embed（也称为“向量化”）这些片段。
        // embed是进行相似性搜索所需的。
        // 在这个例子中，我们将使用本地进程内 embedding 模型，可以选择任何支持的模型。
        // https://huggingface.co/BAAI/bge-small-en-v1.5
        // bge is short for BAAI general embedding.
        // 北京智源人工智能研究院
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // 接下来，我们将这些 embeddings 存储在一个嵌入存储（也称为“向量数据库”）中。
        // 这个存储将在每次与LLM交互时用于搜索相关的段落。
        // 为了简单起见，这个例子使用的是内存中的嵌入存储，但你可以选择任何支持的存储。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // 内容检索器负责根据用户查询检索相关内容。
        // 目前，它能够检索文本片段，但未来的增强功能将包括支持其他模式，如图像、音频等。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2) // 在每次交互中，我们将检索最相关的两个片段
                .minScore(0.5) // 我们希望检索到至少与用户查询有些相似的片段
                .build();


        // 可以使用聊天记忆，与LLM进行来回对话并允许它记住以前的互动。
        // 目前，提供了两种聊天记忆实现：MessageWindowChatMemory 和 TokenWindowChatMemory。
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);


        // 最后一步是构建我们的AI服务，配置它以使用我们上面创建的组件。
        return AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .contentRetriever(contentRetriever)
                .chatMemory(chatMemory)
                .build();
    }
}
