# Welcome to Question Answering Chatbot Powered by the GPT Models 🚀🤖🦜

This chatbot provides a variety of chat profiles to cater to different user needs. Each profile has a unique way of handling and processing user queries.

Here's how you can interact with it:

1. Use the sliders and switches on the left to adjust the settings.
2. Type your query in the input box at the bottom.
3. Press Enter to submit your query.
4. The application will process your query and provide a response with the sources where you can find a link to documents that were used to generate a response.
5. Use the 'Check All Indexes' and 'Uncheck All Indexes' buttons to quickly select or deselect all indexes in the settings.

## Available Chat Profiles

- **Wikis Assistant**: This profile is an extension of the Wikis Q&A. The retrieved documents are used as the context for the entire conversation, providing a more comprehensive response to your queries.
- **SharePoint Assistant**: Used for Q&A with retrieval of chunks of documents from SharePoint. Currently it supports most SharePoint site pages but only some document libraries of SharePoint sites.
- **Agent**: This profile is designed for chatting with tools. It includes web search, Azure DevOps and SharePoint retrievers, and a calculator. This allows for a more interactive and dynamic response to your queries.
- **Summarise a Document**: This profile is used for Q&A with summarization of documents. It currently supports PDFs, DOCXs, PowerPoint, Excel, and text files. This allows for a more concise and summarized response to your queries.
- **General Chat**: This profile is designed for general chatting with GPT models. It provides a more casual and conversational approach to your queries.
- **TEMPLATE Financial 2023**: This profile is used for chatting about TEMPLATE's 2023 financial results. It uses a multimodal RAG that includes tables and images, which are sent to gpt-4-vision for processing and response generation.
- **Multi-Query Wikis Assistant**: This profile is similar to the Wikis Q&A, but with a twist. The retrieved documents are used as the context for the entire conversation, and multiple queries are generated for the retriever. This allows for a more dynamic and interactive Q&A experience.
- **SharePoint Multi-Query Assistant**: This profile is designed for Q&A with retrieval of chunks of documents from SharePoint. The documents are used as the context for the entire conversation, and multiple queries are generated for the retriever. This allows for a more detailed and thorough response to your queries.
- **TEMPLATE Poland Handbook**: This profile is used for Q&A with the Employee handbook for TEMPLATE Poland. It provides a more specific and detailed response to your queries related to the handbook.
- **TEMPLATE Poland Handbook Unstructured**: This profile is used for Q&A with the Employee handbook for TEMPLATE Poland. The handbook is processed with an Unstructured library, providing a more comprehensive and detailed response to your queries.
- **TEMPLATE Poland Handbook Single**: This profile is used for Q&A with the Employee handbook for TEMPLATE Poland. The handbook is processed with an Unstructured library, providing a more focused and specific response to your queries.
- **SharePoint Ensemble**: This profile is used for Q&A with retrieval of chunks of documents from SharePoint. It uses an ensemble of vector search with the BM25 algorithm, providing a more accurate and relevant response to your queries.
- **Wikis Q&A**: This profile is used for Q&A with retrieval of wikis from Azure DevOps and the TEMPLATE Wiki. It provides a more specific and detailed response to your queries related to the wikis.
- **SharePoint Full Docs**: Used for Q&A with retrieval of documents from SharePoint. Currently it supports most SharePoint site pages but only some document libraries of SharePoint sites.

For profiles related to Wikis you can start a conversation here or try these examples:

- **Learn how to set up services** - `How do I create a capability?`
- **Help for new crew members** - `Summarize the following things I should install on my laptop as a TEMPLATE developer`
- **How do I create a Kafka topic?** - `How do I create a Kafka topic?`
- **Docker in TEMPLATE** - `How is Docker used in TEMPLATE?`
- **Kubernetes at TEMPLATE** - `What is the role of Kubernetes in TEMPLATE's infrastructure?`
- **React usage** - `How is React utilized in TEMPLATE's front-end development?`
- **Python applications** - `What kind of applications are developed using Python in TEMPLATE?`
- **Go in TEMPLATE** - `How is Go language used in TEMPLATE's services?`
- **JavaScript in TEMPLATE** - `What role does JavaScript play in TEMPLATE's web development?`
- **Understanding TEMPLATE's CI/CD pipeline** - `How is the CI/CD pipeline configured in TEMPLATE?`
- **TEMPLATE's cloud strategy** - `What cloud platforms are used in TEMPLATE and why?`
- **Microservices at TEMPLATE** - `How does TEMPLATE design and manage microservices?`

For profiles related to SharePoint, you can start a conversation here or try these examples:

- **Company Benefits** - `What are my company benefits?`
- **Holiday Policy in Denmark** - `How many weeks of holiday do I have if I'm in Denmark?`
- **Company Policies** - `What are the company policies at TEMPLATE?`
- **Working Hours** - `What are the standard working hours at TEMPLATE?`
- **Sick Leave Policy** - `What is the sick leave policy at TEMPLATE?`
- **Maternity Leave in Denmark** - `What is the maternity leave policy in Denmark?`
- **Employee Benefits in Denmark** - `What are the employee benefits for TEMPLATE employees in Denmark?`