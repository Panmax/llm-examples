package me.jiapan.examples.llm.tools;

import me.jiapan.examples.llm.shared.Assistant;
import me.jiapan.examples.llm.shared.Utils;
import dev.langchain4j.code.judge0.Judge0JavaScriptExecutionTool;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;

import static java.time.Duration.ofSeconds;

public class Example2_DynamicTools {

    public static void main(String[] args) {

        Judge0JavaScriptExecutionTool judge0Tool = new Judge0JavaScriptExecutionTool(Utils.RAPID_API_KEY);

        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .temperature(0.0)
                .timeout(ofSeconds(60))
                .logRequests(true) // NOTE: 可以解开注释看看请求和响应的日志
                .logResponses(true)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                // .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
                .tools(judge0Tool)
                .build();

        interact(assistant, "What is the square root of 49506838032859?");
        interact(assistant, "Capitalize every third letter: abcabc");
        interact(assistant, "What is the number of hours between 17:00 on 21 Feb 1988 and 04:00 on 12 Apr 2014?");
    }

    private static void interact(Assistant assistant, String userMessage) {
        System.out.println("[User]: " + userMessage);
        String answer = assistant.answer(userMessage);
        System.out.println("[Assistant]: " + answer);
        System.out.println();
        System.out.println();
    }
}
