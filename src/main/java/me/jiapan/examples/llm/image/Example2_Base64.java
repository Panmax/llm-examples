package me.jiapan.examples.llm.image;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.output.Response;
import me.jiapan.examples.llm.shared.Utils;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Base64;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O;

public class Example2_Base64 {
    public static void main(String[] args) throws IOException {
        byte[] imageBytes = Files.readAllBytes(Utils.toPath("PNG_transparency_demonstration_1.png"));
        String base64String = Base64.getEncoder().encodeToString(imageBytes);

        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                // .logRequests(true)
                // .logResponses(true)
                .modelName(GPT_4_O)
                .maxTokens(50)
                .build();

        UserMessage userMessage = UserMessage.from(
                TextContent.from("What do you see? in Chinese"),
                ImageContent.from(base64String, "image/png")
        );
        Response<AiMessage> response = model.generate(userMessage);
        System.out.println(response.content().text());
        // 我看到的是四个骰子，它们的颜色分别是蓝色、红色和绿色。每个骰子上都有白色的点数。背景有彩色的模糊条纹。
        System.out.printf("token usage input:%d, output:%d, total:%d\n",
                response.tokenUsage().inputTokenCount(), response.tokenUsage().outputTokenCount(), response.tokenUsage().totalTokenCount());
        // token usage input:99, output:44, total:143
    }
}
