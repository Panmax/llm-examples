package me.jiapan.examples.llm.shared;

/**
 * 这是一个“AiService”。它是一个具有AI功能 / 特性的Java服务。
 * 它可以像任何其他服务一样集成到您的代码中，作为bean，并且可以模拟进行测试。
 * 目标是将AI功能无缝集成到您（现有）的代码库中，尽量减少摩擦。
 * 它在概念上类似于Spring Data JPA或Retrofit。
 * 您定义一个接口并可选地使用注解自定义它。
 * 然后LangChain4j使用代理和反射为此接口提供实现。
 * 这种方法抽象掉了所有的复杂性和样板代码。
 * 因此，您不需要处理模型、消息、内存、RAG组件、工具、输出解析器等问题。
 * 不过，不用担心。它非常灵活且可配置，因此您能够根据您的具体用例进行定制。
 */
public interface Assistant {

    String answer(String query);
}