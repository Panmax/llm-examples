package me.jiapan.examples.llm.image;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
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

public class Example4_Handwriting {

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


        UserMessage userMessage = UserMessage.from(
                TextContent.from(PROMPT),
                ImageContent.from(base64String, "image/png")
        );
        Response<AiMessage> response = model.generate(
                SystemMessage.from("You are a helpful assistant. Help me with my math homework!"),
                userMessage);
        System.out.println(response.content().text());
        // ```markdown
        // ### Identify the question
        //         $$\frac{3}{10} + \frac{5}{12} =$$
        //
        // ### Describe the student's problem-solving process and the answer in the image
        //         The student attempted to find a common denominator for the fractions $\frac{3}{10}$ and $\frac{5}{12}$. They incorrectly wrote the common denominator as 60 for both fractions without adjusting the numerators accordingly.
        //
        // ### Determine whether the student's problem-solving process and the answer are correct
        //         The student's problem-solving process is incorrect. The correct common denominator for $\frac{3}{10}$ and $\frac{5}{12}$ is 60, but the numerators need to be adjusted as follows:
        //
        //         $$\frac{3}{10} = \frac{3 \times 6}{10 \times 6} = \frac{18}{60}$$
        //         $$\frac{5}{12} = \frac{5 \times 5}{12 \times 5} = \frac{25}{60}$$
        //
        //         So, the correct addition should be:
        //
        //         $$\frac{18}{60} + \frac{25}{60} = \frac{43}{60}$$
        //
        //         The student's answer is incorrect.
        // ```
        System.out.printf("token usage input:%d, output:%d, total:%d\n",
                response.tokenUsage().inputTokenCount(), response.tokenUsage().outputTokenCount(), response.tokenUsage().totalTokenCount());
        // token usage input:246, output:231, total:477
    }
}
