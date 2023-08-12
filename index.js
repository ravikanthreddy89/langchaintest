// langchain imports
const { OpenAI } = require("langchain/llms/openai");
const { OpenAIEmbeddings } =require("langchain/embeddings/openai");
const { TextLoader } = require("langchain/document_loaders/fs/text");
const { CSVLoader } = require("langchain/document_loaders/fs/csv");
const { DirectoryLoader } = require("langchain/document_loaders/fs/directory");
const { JSONLoader, JSONLinesLoader } = require("langchain/document_loaders/fs/json");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { HNSWLib } = require("langchain/vectorstores/hnswlib");
const { RetrievalQAChain }= require("langchain/chains");

// other util imports
const {isEmpty} = require('lodash');

async function readCsvFile() {
    try {
        const loader = new CSVLoader("sample_docs/example.csv");
        const docs = await loader.load();
        //console.log("Successfully loaded csv documents");
        //console.log(docs) 
    } catch (error) {
        console.error("failed to load csv documents");
    }
}
readCsvFile();


/**
 * 
 * @param {path} "Location or path to 'source documents' on filesystem"
 */
async function loadDocsFromDirectory({ path = "sample_docs/" }) {
    try {
        const loader = new DirectoryLoader(path, {
            ".json": (path) => new JSONLoader(path),
            ".txt": (path) => new TextLoader(path),
            ".csv": (path) => new CSVLoader(path),
            ".mdx": (path) => new TextLoader(path)
        });
        const docs = await loader.load();
        //console.log({ docs });
        return docs;
    } catch (error) {
        console.error(error)
    }
}

async function splitAndChunK({docs = []}) {
    if(isEmpty(docs)) {
        console.log("Documents are empty no Split and chunk");
        return;
    }

    const splitter = new RecursiveCharacterTextSplitter();
    const docOutput = await splitter.splitDocuments(docs);
    
    return docOutput
}




async function main() {
    // ingest/load the documents
    const docs = await loadDocsFromDirectory({ path: "langchain_web_docs" });
    const docOutput = await splitAndChunK({docs});
    //console.log(docOutput);
    
    // Create a vector store from the documents using OpenAI embeddings
    const vectorStore = await HNSWLib.fromDocuments(docOutput, new OpenAIEmbeddings({
        openAIApiKey : 'dummykey'
    }));

    // Initialize a retriever wrapper around the vector store
    const vectorStoreRetriever = vectorStore.asRetriever();
    

    const model = new OpenAI({
        openAIApiKey : 'dummykey'
    });
    const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);    
    const res = await chain.call({
        query: "list of langchain transformers",
    });
    console.log({ res });
}

main();

// const llm = new OpenAI({
//     temperature: 0.9
// });

// async function getResult() {
//     const result = await llm.predict("What is a good place to eat biryani in Hyderabad?");
//     console.log(result);
// }
//getResult()