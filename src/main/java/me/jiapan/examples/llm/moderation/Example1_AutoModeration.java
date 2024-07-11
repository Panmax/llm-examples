package me.jiapan.examples.llm.moderation;

import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiModerationModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Moderate;
import dev.langchain4j.service.ModerationException;

public class Example1_AutoModeration {
    interface Chat {
        @Moderate
        String chat(String text);
    }

    // 用户发起提问时，可以同时过一下 moderation，如果不符合规范，则不展示大模型的回答
    public static void main(String[] args) {

        OpenAiModerationModel moderationModel = OpenAiModerationModel.builder()
                .logRequests(true).logResponses(true).apiKey(System.getenv("OPENAI_API_KEY")).build();

        Chat chat = AiServices.builder(Chat.class)
                .chatLanguageModel(OpenAiChatModel
                        .builder()
                        .logRequests(true).logResponses(true)
                        .apiKey(System.getenv("OPENAI_API_KEY")).build())
                .moderationModel(moderationModel)
                .build();

        try {
            System.out.println(chat.chat("I WILL KILL YOU!!!"));;
        } catch (ModerationException e) {
            System.out.println(e.getMessage());
            // Text "I WILL KILL YOU!!!" violates content policy
        }
    }

}
