package me.jiapan.examples.llm.image;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.output.Response;
import org.apache.commons.lang3.time.StopWatch;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Base64;

import static me.jiapan.examples.llm.shared.Utils.toPath;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O;

public class Example3_Base64vsUrl {
    public static void main(String[] args) throws IOException {
        byte[] imageBytes = Files.readAllBytes(toPath("PNG_transparency_demonstration_1.png"));
        String base64String = Base64.getEncoder().encodeToString(imageBytes);

        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                // .logRequests(true)
                // .logResponses(true)
                .modelName(GPT_4_O)
                .maxTokens(50)
                .build();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        int token = 0;
        for (int i = 0; i < 10; i++) {
            UserMessage userMessage = UserMessage.from(
                    TextContent.from("What do you see? in Chinese"),
                    ImageContent.from(base64String, "image/png", ImageContent.DetailLevel.HIGH)
            );
            Response<AiMessage> response = model.generate(userMessage);
            token += response.tokenUsage().inputTokenCount();
        }
        stopWatch.stop();
        System.out.printf("base64 duration %dms, token:%d\n", stopWatch.getTime(), token);
        // base64 duration 30935ms
        // base64 duration 30289ms
        // base64 duration 31595ms
        // base64 duration 30240ms
        // base64 duration 28828ms

        stopWatch.reset();
        stopWatch.start();
        token = 0;
        for (int i = 0; i < 10; i++) {
            UserMessage userMessage = UserMessage.from(
                    TextContent.from("What do you see? in Chinese"),
                    ImageContent.from("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png", ImageContent.DetailLevel.HIGH)
            );
            Response<AiMessage> response = model.generate(userMessage);
            token += response.tokenUsage().inputTokenCount();
        }
        stopWatch.stop();
        System.out.printf("url duration %dms, token:%d\n", stopWatch.getTime(), token);
        // url duration 24615ms
        // url duration 27237ms
        // url duration 28477ms
        // url duration 25713ms
        // url duration 24583ms


        // 结论：
        // - token 消耗无区别
        // - 耗时: base64 比 url 慢10-20%
        // - 考虑到如果要生成url需要先上传到oss，加上这段时间两者差距就不明显了
        // - 建议：单次调用使用base64，多次调用可以考虑使用url
        // - 增量调用时（如用户手写过程）前边的图使用url，最新的图使用base64，并异步上传到oss供下次使用
    }
}
