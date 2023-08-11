const { OpenAI } = require("langchain/llms/openai");

const llm = new OpenAI({
    openAIApiKey: "",
    temperature: 0.9
});

async function getResult() {
    const result = await llm.predict("What is a good place to eat biryani in Hyderabad?");
    console.log(result);
}

getResult()