package me.jiapan.examples.llm.image;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.output.Response;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O;

public class Example1_Easy {
    public static void main(String[] args) {
        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                // .logRequests(true)
                // .logResponses(true)
                .modelName(GPT_4_O)
                // .maxTokens(50)
                .build();

        UserMessage userMessage = UserMessage.from(
                TextContent.from("What do you see? in Chinese"),
                ImageContent.from("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png", ImageContent.DetailLevel.LOW)
        );
        Response<AiMessage> response = model.generate(userMessage);
        System.out.println(response.content().text());
        // 我看到的是四个骰子。它们分别是蓝色、绿色、红色和黄色，每个骰子上都有白色的点数。
        System.out.printf("token usage input:%d, output:%d, total:%d\n\n",
                response.tokenUsage().inputTokenCount(), response.tokenUsage().outputTokenCount(), response.tokenUsage().totalTokenCount());
        // token usage input:99, output:33, total:132

        userMessage = UserMessage.from(
                TextContent.from("What do you see? in Chinese"),
                ImageContent.from("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png", ImageContent.DetailLevel.AUTO)
        );
        response = model.generate(userMessage);
        System.out.println(response.content().text());
        // 我看到四个骰子。它们分别是蓝色、红色、绿色和黄色的，每个骰子上都有白色的点数。
        System.out.printf("token usage input:%d, output:%d, total:%d\n\n",
                response.tokenUsage().inputTokenCount(), response.tokenUsage().outputTokenCount(), response.tokenUsage().totalTokenCount());
        // token usage input:779, output:33, total:812


        userMessage = UserMessage.from(
                TextContent.from("What do you see? in Chinese"),
                ImageContent.from("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png", ImageContent.DetailLevel.HIGH)
        );
        response = model.generate(userMessage);
        System.out.println(response.content().text());
        // 我看到四个色彩鲜艳的骰子。它们分别是红色、绿色、蓝色和黄色的，并且每个骰子上都有白色的点数。背景是五彩斑斓的条
        System.out.printf("token usage input:%d, output:%d, total:%d\n",
                response.tokenUsage().inputTokenCount(), response.tokenUsage().outputTokenCount(), response.tokenUsage().totalTokenCount());
        // token usage input:779, output:50, total:829
    }
}
