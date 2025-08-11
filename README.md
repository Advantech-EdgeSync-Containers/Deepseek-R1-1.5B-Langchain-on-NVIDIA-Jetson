# Deepseek-R1 1.5B Langchain on NVIDIA Jetson™

**Version:** 1.0
**Release Date:** May 2025
**Copyright:** © 2025 Advantech Corporation. All rights reserved.

## Overview
Deepseek-R1 1.5B Langchain on NVIDIA Jetson™ Image delivers a modular, high-performance AI chat solution tailored for Jetson™ edge devices. It combines Ollama with the DeepSeek R1 1.5B model for LLM inference, a FastAPI-based Langchain middleware for orchestration and tool integration, and OpenWebUI for an intuitive user interface. The container supports Retrieval-Augmented Generation (RAG), tool-augmented reasoning, conversational memory, and custom LLM workflows, making it ideal for building intelligent, context-aware agents. It is fully optimized for hardware acceleration on Jetson™ platforms & provides a development environment to build RAGs and AI agent use cases.

## Key Features

| Feature | Description |
|--------|-------------|
| Integrated OpenWebUI | Clean, user-friendly frontend for LLM chat interface |
| DeepSeek R1 1.5B Inference | Efficient on-device LLM via Ollama; minimal memory, high performance |
| Model Customization | Create or fine-tune models using `ollama create` |
| REST API Access | Simple local HTTP API for model interaction |
| Flexible Parameters | Adjust inference with `temperature`, `top_k`, `repeat_penalty`, etc. |
| Modelfile Customization | Configure model behavior with Docker-like `Modelfile` syntax |
| Prompt Templates | Supports formats like `chatml`, `llama`, and more |
| LangChain Integration | Multi-turn memory with `ConversationChain` support |
| FastAPI Middleware | Lightweight interface between OpenWebUI and LangChain |
| Offline Capability | Fully offline after container image setup; no internet required |

## Architecture
![langchain-deepseek.png](..%2Fdata%2Farchitectures%2Flangchain-deepseek.png)

## Repository Structure
```
Deepseek-R1-1.5B-Langchain-on-NVIDIA-Jetson/
├── .env                                      # Environment configuration
├── build.sh                                  # Build helper script
├── wise-bench.sh                             # Wise Bench script
├── docker-compose.yml                        # Docker Compose setup
├── README.md                                 # Overview
├── quantization-readme.md                    # Model quantization steps
├── other-AI-capabilities-readme.md           # Other AI capabilities supported by container image
├── llm-models-performance-notes-readme.md    # Performance notes of LLM Models
├── efficient-prompting-for-compact-models.md # Craft better prompts for small and quantized language models
├── customization-readme.md                   # Customization, optimization & configuration guide
├── .gitignore                                # Git ignore specific files
└── langchain-service/                        # Core LangChain API service
    ├── app.py                                # Main LangChain-FastAPI app
    ├── llm_loader.py                         # LLM loader (Ollama, DeepSeek, etc.)
    ├── requirements.txt                      # Python dependencies
    ├── schema.py                             # Request schema helper
    ├── utils.py                              # Utility functions helper
    └── start_services.sh                     # Startup script
```

## Container Description

### Quick Information

`build.sh` will start following two containers:

| Container Name | Description |
|-----------|---------|
| Deepseek-R1-1.5B-Langchain-on-NVIDIA-Jetson | Provides a hardware-accelerated development environment using various AI software components along with Deepseek R1 1.5B, Ollama & Langchain   |
| openweb-ui-service | Optional, provides UI which is accessible via browser for inferencing |

### Deepseek-R1-1.5B-Langchain-on-NVIDIA-Jetson Container Highlights

This container leverages [**LangChain**](https://www.langchain.com/) as the core orchestration framework for building powerful, modular LLM applications directly on NVIDIA Jetson™ devices. It integrates with the local inference engine Ollama, enabling offline, edge-optimized AI workflows without relying on cloud services.

| Feature                   | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| Middleware Logic Engine   | FastAPI-based LangChain server handles agent logic, tools, memory, and RAG pipelines.    |
| LLM Integration           | Connects to On-device model (Deepseek R1 1.5B) via Ollama.                               |
| RAG-Enabled               | Supports Retrieval-Augmented Generation using vector stores and document loaders.        |
| Agent & Tool Support      | Easily define and run LangChain agents with tool integration (e.g., search, calculator). |
| Conversational Memory     | Includes support for memory modules like buffer, summary, or vector-based recall.        |
| Streaming & Async Support | Real-time response streaming for chat UIs via FastAPI endpoints.                         |
| Offline-First             | All components run locally after model download—ensures low latency and data privacy.    |
| Modular Architecture      | Plug-and-play design with support for custom chains, tools, and prompts.                 |
| Developer Friendly        | Exposes RESTful APIs; works with OpenWebUI, custom frontends, or CLI tools.              |
| Hardware Accelerated      | Optimized for Jetson™ devices using quantized models and accelerated inference.          |

### OpenWebUI Container Highlights

OpenWebUI serves as a clean and responsive frontend interface for interacting with LLMs via APIs like Ollama or OpenAI-compatible endpoints. When containerized, it provides a modular, portable, and easily deployable chat interface suitable for local or edge deployments.

| Feature                         | Description                                                     |
|---------------------------------|-----------------------------------------------------------------|
| User-Friendly Interface         | Sleek, chat-style UI for real-time interaction.                 |
| OpenAI-Compatible Backend       | Works with Ollama, OpenAI, and similar APIs with minimal setup. |
| Container-Ready Design          | Lightweight and optimized for edge or cloud deployments.        |
| Streaming Support               | Enables real-time response streaming for interactive UX.        |
| Authentication & Access Control | Basic user management for secure access.                        |
| Offline Operation               | Runs fully offline with local backends like Ollama.             |

## List of READMEs

| Module                       | Link                                                  | Description                                                                |
|------------------------------|-------------------------------------------------------|----------------------------------------------------------------------------|
| Quick Start                  | [README](./README.md)                                 | Overview of the container image                                            |
| Customization & optimization | [README](./customization-readme.md)                   | Steps to customize a model, configure environment, and optimize            |
| Model Performances           | [README](./llm-models-performance-notes-readme.md)    | Performance stats of various LLM Models                                    |
| Other AI Capabilities        | [README](./other-AI-capabilities-readme.md)           | Other AI capabilities supported by the container                           |
| Quantization                 | [README](./quantization-readme.md)                    | Steps to quantize a model                                                  |
| Prompt Guidelines            | [README](./efficient-prompting-for-compact-models.md) | Guidelines to craft better prompts for small and quantized language models |

## Model Information  

This image uses DeepSeek R1-1.5B for inferencing; here are the details about the model used:

| Item                                                   | Description                     |
|--------------------------------------------------------|---------------------------------|
| Model source                                           | Ollama Model (deepseek-r1:1.5b) |
| Model architecture                                     | Qwen2                           |
| Model quantization                                     | Q4_K_M                          |
| Ollama command                                         | ollama pull deepseek-r1:1.5b    |
| Number of Parameters                                   | ~1.78 B                         |
| Model size                                             | ~1.1 GB                         |
| Default context size (unless changed using parameters) | 2048                            |

## Hardware Specifications

| Component       | Specification                                     |
|-----------------|---------------------------------------------------|
| Target Hardware | NVIDIA Jetson™                                    |
| GPU             | NVIDIA® Ampere architecture with 1024 CUDA® cores |
| DLA Cores       | 1 (Deep Learning Accelerator)                     |
| Memory          | 4/8/16 GB shared GPU/CPU memory                   |
| JetPack Version | 5.x                                               |

## Software Components

The following software components are available in the base image:

| Component    | Version        | Description                        |
|--------------|----------------|------------------------------------|
| CUDA®        | 11.4.315       | GPU computing platform             |
| cuDNN        | 8.6.0          | Deep Neural Network library        |
| TensorRT™    | 8.5.2.2        | Inference optimizer and runtime    |
| PyTorch      | 2.0.0+nv23.02  | Deep learning framework            |
| TensorFlow   | 2.12.0 | Machine learning framework         |
| ONNX Runtime | 1.16.3         | Cross-platform inference engine    |
| OpenCV       | 4.5.0          | Computer vision library with CUDA® |
| GStreamer    | 1.16.2         | Multimedia framework               |


The following software components/packages are provided further inside the container image:

| Component        | Version     | Description                                                             |
|------------------|-------------|-------------------------------------------------------------------------|
| Ollama           | 0.5.7       | LLM inference engine                                                    |
| LangChain        | 0.2.17      | Installed via PIP, framework to build LLM applications                  |
| FastAPI          | 0.115.12    | Installed via PIP, develop OpenAI-compatible APIs for serving LangChain |
| OpenWebUI        | 0.6.5       | Provided via separate OpenWebUI container for UI                        |
| DeepSeek R1 1.5B | N/A         | Pulled inside the container and persisted via docker volume             |
| FAISS            | 1.8.0.post1 | Vector store backend for enabling RAG with efficient similarity search  |

## Quick Start

### Installation
```
# Clone the repository
git clone https://github.com/Advantech-EdgeSync-Containers/Nagarro-Container-Project.git
cd Deepseek-R1-1.5B-Langchain-on-NVIDIA-Jetson

# Make the build script executable
chmod +x build.sh

# Launch the container
sudo ./build.sh

```

### Run Services

After installation succeeds, by default control lands inside the container. Run the following command to start services within the container.

```
# Under /workspace/langchain-service, run this command
# Provide executable rights
chmod +x start_services.sh

# Start services
./start_services.sh
```
Allow some time for the OpenWebUI and Deepseek-R1 1.5B Langchain on NVIDIA Jetson™ to settle and become healthy.

### AI Accelerator and Software Stack Verification (Optional)
```
# Verify AI Accelerator and Software Stack Inside Docker Container
/workspace/wise-bench.sh
```
![langchain-wise-bench.png](..%2Fdata%2Fimages%2Flangchain-wise-bench.png)

Wise-bench logs are saved in `wise-bench.log` file under `/workspace`

### Check Installation Status
Exit from the container and run the following command to check the status of the containers:
```
sudo docker ps
```
Allow some time for containers to become healthy.

### UI Access
Access OpenWebUI via any browser using the URL given below. Create an account and perform a login:
```
http://localhost_or_Jetson_IP:3000
```
### Select Model
In case Ollama has multiple models available, choose from the list of models on the top-left of OpenWebUI after signing up/logging in successfully. As shown below. Select DeepSeek R1 1.5B:

![Select Model](..%2Fdata%2Fimages%2Fselect-model.png)

### Quick Demonstration:

![Demo](..%2Fdata%2Fgifs%2Flangchain-deepseek.gif)

## Prompt Guidelines

This [README](./efficient-prompting-for-compact-models.md) provides essential prompt guidelines to help you get accurate and reliable outputs from small and quantized language models.

## Ollama Logs and Troubleshooting

### Log Files

Once services have been started inside the container, the following log files are generated:

| Log File | Description |
|-----------|---------|
| ollama.pid | Provides process-id for the currently running Ollama service   |
| ollama.log | Provides Ollama service logs |
| uvicorn.log | Provides FastAPI-Langchain service logs |
| uvicorn.pid | Provides FastAPI-Langchain service pid |

### Troubleshoot

Here are quick commands/instructions to troubleshoot issues with the Jetson™ Deepseek Langchain Container:

- View service logs within the container
  ```
  tail -f ollama.log # or
  tail -f uvicorn.log
  ```

- Check if the model is loaded using CPU or GPU or partially both (ideally, it should be 100% GPU loaded).
  ```
  ollama ps
  ```

- Kill & restart services within the container (check pid manually via `ps -eaf` or use pid stored in `ollama.pid` or `uvicorn.pid`)
  ```
  kill $(cat ollama.pid)
  kill $(cat uvicorn.pid)
  ./start_services.sh
  ```

  Confirm there is no Ollama & FastAPI service running using:
  ```
  ps -eaf
  ```

- Enable debug mode for the Ollama service (kill the existing Ollama service first).
  ```
  export OLLAMA_DEBUG=true
  ./start_services.sh
  ```

- In some cases, it has been found that if Ollama is also present at the host, it may give permission issues during pulling models within the container. Uninstalling host Ollama may solve the issue quickly. Follow this link for uninstallation steps - [Uninstall Ollama.](https://github.com/ollama/ollama/blob/main/docs/linux.md#uninstall)


## Best Practices and Recommendations

### Memory Management & Speed
- Ensure models are fully loaded into GPU memory for best results.
- Batch inference for better throughput
- Use stream processing for continuous data
- Offload unwanted models from GPU (use the Keep-Alive parameter for customizing this behavior).
- Enable Jetson™ Clocks for better inference speed
- Used quantized models to balance speed and accuracy
- Increase swap size if models loaded are large
- Use lesser context & batch size to avoid high memory utilization
- Set max-tokens in API payloads to avoid unnecessarily long response generations, which may affect memory utilization.

### Ollama Model Behavior Corrections 
- Restart Ollama services
- Remove the model once and pull it again
- Check if the model is correctly loaded into the GPU or not; it should show loaded as 100% GPU. 
- Create a new Modelfile and set parameters like temperature, repeat penalty, system, etc., as needed to get expected results.

### LangChain Middleware Tuning
- Use asynchronous chains and streaming response handlers to reduce latency in FastAPI endpoints.
- For RAG pipelines, use efficient vector stores (e.g., FAISS with cosine or inner product) and pre-filter data when possible; prefer models with ~3B parameters for better context reliability.
- Avoid long chain dependencies; break workflows into smaller composable components.
- Cache prompt templates and tool results when applicable to reduce unnecessary recomputation
- For agent-based flows, limit tool calls per loop to avoid runaway execution or high memory usage.
- Log intermediate steps (using LangChain’s callbacks) for better debugging and observability
- Use models with ≥3B parameters (e.g., Llama 3.2 3B or larger) for agent development to ensure better reasoning depth and tool usage reliability.

## REST API Access

[**Official Documentation**](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Ollama APIs
Ollama APIs are accessible on the default endpoint (unless modified). If needed, APIs could be called using code or curl as below:

Inference Request:
```
curl http://localhost_or_Jetson_IP:11434/api/generate -d '{
  "model": "deepseek-r1:1.5b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```
Here stream mode could be changed to true/false as per the needs.

Response:
```
{
  "model": "deepseek-r1:1.5b",
  "created_at": "2023-08-04T08:52:19.385406455-07:00",
  "response": "<HERE_WILL_THE_RESPONSE>",
  "done": false
}
```
Sample Screenshot:

![ollama-curl.png](..%2Fdata%2Fimages%2Follama-curl.png)

For further API details, please refer to the official documentation of Ollama as mentioned on top.

### FastAPI (Serving LangChain)
Swagger docs could be accessed on the following endpoint:
```
http://localhost_or_Jetson_IP:8000/docs
```
Sample Screenshot:

![fast-api.png](..%2Fdata%2Fimages%2Ffast-api.png)

Inference Request:
```
curl -X 'POST' \
  'http://localhost_or_Jetson_IP:8000/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "string",
  "messages": [
    {
      "role": "user",
      "content": "Hi"
    }
  ],
  "stream": true
}'
```
Response:
```
data: {"id": "992f00ed-5c75-4d9e-b177-3a4a815044e1", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "<think>"}, "index": 0, "finish_reason": null}]}
data: {"id": "594dc272-7d2a-4bdd-8020-4ecb6a618e1a", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "\n\n"}, "index": 0, "finish_reason": null}]}
data: {"id": "5a0e84ce-3cb8-47bb-9d79-b3049f07fe5e", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "</think>"}, "index": 0, "finish_reason": null}]}
data: {"id": "88f7035d-aa87-4b7b-bb43-111675bd2bf4", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "\n\n"}, "index": 0, "finish_reason": null}]}
data: {"id": "247efa16-9312-4365-ba86-caf1a8eeba0a", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Hello"}, "index": 0, "finish_reason": null}]}
data: {"id": "511c7066-81c2-435b-b6ee-a5867bbf4278", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "!"}, "index": 0, "finish_reason": null}]}
data: {"id": "f5d5d7dd-c2fd-48d0-a523-453ad949f9f3", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " I"}, "index": 0, "finish_reason": null}]}
data: {"id": "d1030af7-42e5-4364-90f2-977b3b798881", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "'m"}, "index": 0, "finish_reason": null}]}
data: {"id": "1ac9f459-0412-41c6-97f8-4042c38e412a", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " Deep"}, "index": 0, "finish_reason": null}]}
data: {"id": "da00b6d6-2293-4f36-8f7d-c22761940b47", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Seek"}, "index": 0, "finish_reason": null}]}
data: {"id": "aadb2200-144f-415e-9bb5-395cde6b2cc8", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "-R"}, "index": 0, "finish_reason": null}]}
data: {"id": "28a645a7-be76-4d7f-9a1c-bf53130aa771", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "1"}, "index": 0, "finish_reason": null}]}
data: {"id": "60568346-3b34-42e1-9d4f-55a6acf562df", "object": "chat.completion.chunk", "choices": [{"delta": {"content": ","}, "index": 0, "finish_reason": null}]}
data: {"id": "5b873b14-5021-4e27-9894-e1d586452591", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " an"}, "index": 0, "finish_reason": null}]}
data: {"id": "4d405a40-c4c0-41cc-8077-ea1f90de384f", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " artificial"}, "index": 0, "finish_reason": null}]}
data: {"id": "f57bf86b-7496-4f51-9c8c-bc17b2c0f94c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " intelligence"}, "index": 0, "finish_reason": null}]}
data: {"id": "b7186ff7-d6ea-4b8f-9ba7-47e4130ca6bb", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " assistant"}, "index": 0, "finish_reason": null}]}
data: {"id": "e61a3cf2-5f39-4092-b620-87a8c4690d9d", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " created"}, "index": 0, "finish_reason": null}]}
data: {"id": "63fa140f-275b-422e-80b3-95637435dd52", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " by"}, "index": 0, "finish_reason": null}]}
data: {"id": "02d70086-c348-4fa9-8f39-ecdae12f5536", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " Deep"}, "index": 0, "finish_reason": null}]}
data: {"id": "5e74e1f0-9905-436f-9b7a-338db2697379", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Seek"}, "index": 0, "finish_reason": null}]}
data: {"id": "2260ad12-047d-4659-913b-62d0797c71da", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "."}, "index": 0, "finish_reason": null}]}
data: {"id": "d640f37f-47fb-4d4a-95ec-4b35b7d1090e", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " For"}, "index": 0, "finish_reason": null}]}
data: {"id": "4c62111c-9cca-4688-bc13-35c279f11d3f", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " comprehensive"}, "index": 0, "finish_reason": null}]}
data: {"id": "1e813032-7425-4561-b0f2-23ae02eb4c08", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " details"}, "index": 0, "finish_reason": null}]}
data: {"id": "14842ffb-f597-4249-ae9a-6b9f26e5ad88", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " about"}, "index": 0, "finish_reason": null}]}
data: {"id": "a7ae30a5-aec6-4869-a268-8e2fcf820a13", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " our"}, "index": 0, "finish_reason": null}]}
data: {"id": "63cb565f-ef72-4699-963b-e30cde223d23", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " models"}, "index": 0, "finish_reason": null}]}
data: {"id": "bcb722fc-93aa-4a33-8159-f8af651daf8b", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " and"}, "index": 0, "finish_reason": null}]}
data: {"id": "4a37b230-4ae8-489f-b2c3-6ce8e380ab6d", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " products"}, "index": 0, "finish_reason": null}]}
data: {"id": "e3f590c7-a91d-48e0-ad9b-775c0dddf5e5", "object": "chat.completion.chunk", "choices": [{"delta": {"content": ","}, "index": 0, "finish_reason": null}]}
data: {"id": "e8a23c36-6799-479e-a193-02403de6bf16", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " we"}, "index": 0, "finish_reason": null}]}
data: {"id": "888aa4f4-f0e8-4bd8-b1d0-f5ae96d564bf", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " invite"}, "index": 0, "finish_reason": null}]}
data: {"id": "d0280f2b-3790-4795-b69d-f1fc76f6b2cf", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " you"}, "index": 0, "finish_reason": null}]}
data: {"id": "538f2658-634b-41f2-9ca6-9f7eaf461e6d", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " to"}, "index": 0, "finish_reason": null}]}
data: {"id": "3184150d-9293-4179-8a55-8e7688f27766", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " consult"}, "index": 0, "finish_reason": null}]}
data: {"id": "d53986d8-57e8-4e1f-a2eb-b333e2fd38f8", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " our"}, "index": 0, "finish_reason": null}]}
data: {"id": "aee0a44c-498b-4411-a349-682d05d75507", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " official"}, "index": 0, "finish_reason": null}]}
data: {"id": "997abcb5-a02b-4842-ad62-79da2ca9cd09", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " documentation"}, "index": 0, "finish_reason": null}]}
data: {"id": "0754314c-2953-435f-9a25-71b7ba4fbc95", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "."}, "index": 0, "finish_reason": null}]}
data: [DONE]
```
Please note that the inference response will be in streaming mode only in the case of FastAPI.

Sample Screenshot:

![fast-api-curl.png](..%2Fdata%2Fimages%2Ffast-api-curl.png)

The same requests can also be made from Fast-API swagger docs.

## Known Limitations

1. Execution Time: The model, when inferred for the first time via OpenWebUI, takes longer time (within 10 seconds) as the model gets loaded into the GPU. 
2. RAM Utilization: RAM utilization for running this container image occupies approximately >5 GB RAM when running on NVIDIA® Orin™ NX – 8 GB. Running this image on Jetson™ Nano may require some additional steps, like increasing swap size or using lower quantization as suited. 
3. OpenWebUI Dependencies: When OpenWebUI is started for the first time, it installs a few dependencies that are then persisted in the associated Docker volume. Allow it some time to set up these dependencies. This is a one-time activity.


## Possible Use Cases

Leverage the container image to build interesting use cases like:

- Predictive Maintenance Chatbots: Integrate with edge telemetry or logs to summarize anomalies, explain error codes, or recommend corrective actions using historical context.

- Compliance and Audit Q&A: Run offline LLMs trained on local policy or compliance data to assist with audits or generate summaries of regulatory alignment—ensuring data never leaves the premises.

- Safety Manual Conversational Agents: Deploy LLMs to provide instant answers from on-site safety manuals or procedures, reducing downtime and improving adherence to protocols.

- Technician Support Bots: Field service engineers can interact with the bot to troubleshoot equipment based on past repair logs, parts catalogs, and service manuals.

- Smart Edge Controllers: LLMs can translate human intent (e.g., “reduce line 2 speed by 10%”) into control commands for industrial PLCs or middleware using AI agents.

- Conversational Retrieval (RAG): Integrate with vector databases (like FAISS and ChromaDB) to retrieve relevant context from local documents and enable conversational Q&A over your custom data.

- Tool-Enabled Agents: Create intelligent agents that use calculators, APIs, or search tools as part of their reasoning process—LangChain handles the logic and LLM interface.

- Factory Incident Reporting: Ingest logs or voice input → extract incident type → summarize → trigger automated alerts or next steps

Copyright © 2025 Advantech Corporation. All rights reserved.