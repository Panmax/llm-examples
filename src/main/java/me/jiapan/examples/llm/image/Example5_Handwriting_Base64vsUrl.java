package me.jiapan.examples.llm.image;

import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import me.jiapan.examples.llm.shared.Utils;
import org.apache.commons.lang3.time.StopWatch;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Base64;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O;

public class Example5_Handwriting_Base64vsUrl {

    private static final String PROMPT = "You are a teacher grading a quiz.\n" +
            "\n" +
            "The image contains a question, the student's problem-solving process and the answer. Your task is to identify the problem-solving process and the answer, and to determine whether they are correct.\n" +
            "\n" +
            "\n" +
            "<< FORMATTING >>\n" +
            "Return a markdown code snippet formatted to look like:\n" +
            "```markdown\n" +
            "### identify the question\n" +
            "\n" +
            "### describe the student's problem-solving process and the answer in the image\n" +
            "\n" +
            "### determine whether the student's problem-solving process and the answer are correct\n" +
            "\n" +
            "```\n" +
            "<< OUTPUT (must include ```markdown at the start of the response) >>\n" +
            "<< OUTPUT (must end with ```) >>\n" +
            "<< OUTPUT (must use $ at the beginning and end of inline formulas) >>\n" +
            "<< OUTPUT (must use $$ at the beginning and end of interline formulas) >>";

    public static void main(String[] args) throws IOException {
        byte[] imageBytes = Files.readAllBytes(Utils.toPath("20.png"));
        String base64String = Base64.getEncoder().encodeToString(imageBytes);

        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName(GPT_4_O)
                .temperature(0.0)
                .build();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int i = 0; i < 10; i++) {
            UserMessage userMessage = UserMessage.from(
                    TextContent.from(PROMPT),
                    ImageContent.from(base64String, "image/png")
            );
            model.generate(userMessage);
        }
        stopWatch.stop();
        System.out.printf("base64 duration %dms\n", stopWatch.getTime());
        // base64 duration 62215ms
        // base64 duration 59404ms

        stopWatch.reset();
        stopWatch.start();
        for (int i = 0; i < 10; i++) {
            UserMessage userMessage = UserMessage.from(
                    TextContent.from(PROMPT),
                    ImageContent.from("https://epoch.cretacontent.com/misc/static/1720518543834_20.png")
            );
            model.generate(
                    SystemMessage.from("You are a helpful assistant. Help me with my math homework!"),
                    userMessage);
        }
        stopWatch.stop();
        System.out.printf("url duration %dms\n", stopWatch.getTime());
        // url duration 48805ms
        // url duration 54745ms
    }
}
