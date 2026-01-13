# Intro

This report is structured in the way solution was developed: iteration by iteration, where I will discuss with which ideas I've started and what I've achieved in the final result. All code for iterations is present in the `/notebooks`.

My environment consisted of Google Colab with Pro subscription and I had access to A100 GPU during this assignment. 

# Iteration 1

Model: Qwen/Qwen2.5-32B-Instruct (4-bit)
Approach: Baseline zero-shot prompting
Score: 0.522

In this iteration, I've just decided to start with some baseline model and get a score I can achieve without touching anything, just a model and basic prompt. 

For the model I selected Qwen/Qwen2.5-32B-Instruct, because it fits resources I have and I worked with it previously, knowing that it can handle Ukrainian on the OK level. 

I did not add CoT reasoning here and did not choose a model with thinking capabilities.

I limited the number of new tokens to 5 to get only the letter and set do_sample=False for maximum determinism.

As a result, the first submission achieved accuracy of 0.522.

# Iteration 2

Model: Qwen/Qwen3-30B-A3B-Instruct-2507 (4-bit)
Approach: Zero-shot prompting with newer model
Score: 0.520

The next iteration was built on the more advanced model Qwen/Qwen3-30B-A3B-Instruct-2507. Strangely it has provided the worser (nearly same) result then the more old one model, meanwhile being more technically sophisticated. The received score is 0.520. No significant changes in the approach.

# Iteration 3

Model: Qwen2.5-32B-Instruct + LoRA (Unsloth)
Approach: SFT fine-tuning on ZNO training data
Score: 0.513/0.511

In this iteration I decided to try a fine-tuning approach. I used LoRA via the Unsloth library for efficient fine-tuning of the unsloth/Qwen2.5-32B-Instruct-bnb-4bit model.

Training parameters: r=16, lora_alpha=16, 60 steps, learning rate 4e-5, batch size 2 with gradient accumulation 4.

The model was trained on ZNO training data in a simple instruction-following format: system prompt + question + correct answer letter.

The result turned out worse than baseline - accuracy 0.513 (tried to decrease learning rate 5e-5 and received 0.511).

Seems fine tuning on the such small dataset doesn't make sense. There is no enough data in this dataset, it pollutes model weights by learning format rather then learning new data. 

# Iteration 4

Model: Qwen/Qwen2.5-32B-Instruct
Approach: RAG with FAISS index (93 PDFs)
Score: Local evaluation only

Here I started experimenting with RAG. The idea was to give the model access to relevant textbook materials during inference.

I built a FAISS index from 93 PDF files (textbooks for grades 1-11). I used multilingual-e5-small for embeddings. This resulted in 49,466 chunks.

For each question, the retriever searches for top-3 most relevant documents and adds them to the prompt context. Model — Qwen/Qwen2.5-32B-Instruct.

Testing was done locally on train data, without Kaggle submission.

The decision to feed text books as is was not greatest, many of them contained similar tasks, contained indexes and broken characters overall, so this polluted the context and it was not useful.   

# Iteration 5

Model: Qwen/Qwen3-4B-Instruct-2507
Approach: Improved RAG with text cleaning and chunk filtering
Score: Local evaluation only

Continued developing the RAG approach. Switched to a smaller model Qwen/Qwen3-4B-Instruct-2507 for faster inference. Added text cleaning by removing special characters, normalizing whitespace. Chunk filtering by discarding chunks that are too short or contain little text (less than 20 letters). Used a larger embedding model intfloat/multilingual-e5-base. Chunk size remained 1000 with overlap 200.

Evaluation was done locally. Result still was poor. I've change a PDF from textbooks with exercises, to just big monographies on the ukrainian grammar, literature and history. They still required preprocessing and I didn't have time for that.

# Iteration 6-7

Model: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
Approach: RAG with Ukrainian Wikipedia (ubertext2.0)
Score: 0.555

Changed the data source for RAG - instead of PDFs, I used the Ukrainian Wikipedia corpus (ubertext2.0).

Main changes:
- Used ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2
- Increased chunk size to 4000 characters (overlap 400) for better context
- Added chunk caching to a pickle file for faster restarts
- Batch size 256 for embeddings (faster index building)

Switched back to intfloat/multilingual-e5-small to speed up process.

# Conclusion

The best result I achieved was 0.555. Neither fine-tuning nor RAG managed to significantly improve the simplest iteration 1 approach.

I chose Qwen2.5-32B because it handles Ukrainian better than alternatives. Fine-tuning made things worse (0.513 vs 0.522). With only 3,063 samples, the model learned the dataset format rather than new knowledge. The ZNO training data is simply too small to meaningfully improve a 32B model's understanding of Ukrainian history and literature.

RAG also didn't help in local testing. The textbook PDFs were noisy and full of exercises, indexes, and OCR artifacts. Wikipedia was cleaner but still the retriever often pulled irrelevant context. The embedding model may not capture subtle semantic differences in Ukrainian historical terms well enough.

For the final iteration 7, I processed the full Ukrainian Wikipedia corpus - about 4.4 GB of text. After chunking with 4000 character size and 400 overlap, I got 918,276 chunks. Filtering removed chunks with less than 100 characters or fewer than 20 letters, leaving 899,246 valid chunks (97.9%). Building the FAISS index took around 20 minutes with batch size 256 for embeddings. I used only TOP_K=1 for retrieval - just one chunk per question. The system prompt was in Ukrainian, asking the model to act as a ZNO specialist. Generation was set to max_new_tokens=1 with do_sample=False for determinism. Answer parsing used regex to extract the first Cyrillic letter, defaulting to "А" if nothing found.

Based on my research, the most promising direction would be generating synthetic training data using a stronger model like GPT-4 to create thousands of ZNO-style questions. This could address the data scarcity issue. Another option is DPO to explicitly teach the model to avoid plausible-sounding but wrong answers, many ZNO questions contain such "trick" distractors. Also, using a better embedding model or fine-tuning one on Ukrainian educational content could improve retrieval quality.
