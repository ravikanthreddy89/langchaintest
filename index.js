// langchain imports
const { OpenAI } = require("langchain/llms/openai");
const { OpenAIEmbeddings } =require("langchain/embeddings/openai");
const { TextLoader } = require("langchain/document_loaders/fs/text");
const { CSVLoader } = require("langchain/document_loaders/fs/csv");
const { DirectoryLoader } = require("langchain/document_loaders/fs/directory");
const { JSONLoader, JSONLinesLoader } = require("langchain/document_loaders/fs/json");
const { PDFLoader } = require("langchain/document_loaders/fs/pdf");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { HNSWLib } = require("langchain/vectorstores/hnswlib");
const { RetrievalQAChain }= require("langchain/chains");

// other util imports
const {isEmpty} = require('lodash');

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
            ".mdx": (path) => new TextLoader(path),
            ".pdf": (path) => new PDFLoader(path)
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

// util function to invoke chain and log the result
async function queryWithChain({ chain , query}) {
    const res = await chain.call({
        query
    });
    console.log({res})
}


async function main({path = "sample_docs"}) {
    // ingest/load the documents
    const docs = await loadDocsFromDirectory({ path });
    const docOutput = await splitAndChunK({docs});
    //console.log(docOutput);
    
    // Create a vector store from the documents using OpenAI embeddings
    const vectorStore = await HNSWLib.fromDocuments(docOutput, new OpenAIEmbeddings({
        openAIApiKey : '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'
    }));

    // Initialize a retriever wrapper around the vector store
    const vectorStoreRetriever = vectorStore.asRetriever();
    

    const model = new OpenAI({
        openAIApiKey : 'dummyKey'
    });
    const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);
    queryWithChain({
        chain,
        query: "list of langchain transformers" 
    });
    
    queryWithChain({
        chain,
        query: "explain TransChain transformer",
    });
    
    queryWithChain({
        chain,
        query: "how to ingest pdf docs",
    });

    queryWithChain({
        chain,
        query: "can I use langchain with a local llm model ?",
    });
    
}



main({path: "langchain_pdf_docs"});

// const llm = new OpenAI({
//     temperature: 0.9
// });

// async function getResult() {
//     const result = await llm.predict("What is a good place to eat biryani in Hyderabad?");
//     console.log(result);
// }
//getResult()