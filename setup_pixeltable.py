# Standard library imports
import uuid
from datetime import datetime

# Third-party library imports
from dotenv import load_dotenv

# Pixeltable core imports
import pixeltable as pxt

# Pixeltable function imports - organized by category
# - Image and video processing
from pixeltable.functions import image as pxt_image
from pixeltable.functions.video import extract_audio

# - LLM and AI model integrations
from pixeltable.functions.anthropic import invoke_tools, messages
from pixeltable.functions.huggingface import sentence_transformer, clip
from pixeltable.functions import openai
from pixeltable.functions.mistralai import chat_completions as mistral_chat

# - Data transformation tools
from pixeltable.iterators import (
    DocumentSplitter,
    FrameIterator,
    AudioSplitter,
    StringSplitter,
)
from pixeltable.functions import string as pxt_str

# Custom function imports
import functions

# Load environment variables
load_dotenv()

# Import centralized configuration
import config

# Initialize the app structure - Pixeltable organizes data in directories similar to a file system
# This provides a clean, hierarchical organization for related tables

# WARNING: The following line will DELETE ALL DATA in the 'agents' directory.
pxt.drop_dir("agents", force=True)
pxt.create_dir("agents", if_exists="ignore")  # Use if_exists='ignore' for safety

# DOCUMENT PROCESSING
# ===================
# Create a table to store documents - Pixeltable natively supports document files
documents = pxt.create_table(
    "agents.collection",
    {"document": pxt.Document, "uuid": pxt.String, "timestamp": pxt.Timestamp, "user_id": pxt.String},
    if_exists="ignore",
)
print("Created/Loaded 'agents.collection' table")

# Create a view to chunk documents - Views are computed on-demand and don't duplicate storage
# This is a key Pixeltable feature for efficient data processing
chunks = pxt.create_view(
    "agents.chunks",
    documents,
    iterator=DocumentSplitter.create(  # Iterators transform data automatically
        document=documents.document,
        separators="paragraph",
        metadata="title, heading, page" # Include metadata
    ),
    if_exists="ignore",
)

# Add an embedding index - Pixeltable provides vector search capabilities out of the box
# This enables semantic search across your documents
chunks.add_embedding_index(
    "text",  # The column to index
    string_embed=sentence_transformer.using(
        model_id=config.EMBEDDING_MODEL_ID
    ),  # Use config
    if_exists="ignore",
)


# Define a search query - Pixeltable allows defining reusable queries as functions
# These capture complex logic in a reusable format
@pxt.query  # This decorator registers a query function with Pixeltable
def search_documents(query_text: str, user_id: str):
    # Vector similarity search is built right into Pixeltable's query syntax
    sim = chunks.text.similarity(query_text)  # Calculate semantic similarity
    return (
        chunks.where(
            (chunks.user_id == user_id)
            & (sim > 0.5)
            & (pxt_str.len(chunks.text) > 30)
        )
        .order_by(sim, asc=False)  # SQL-like syntax for complex operations
        .select(
            chunks.text,
            source_doc=chunks.document,  # Select the source document column
            sim=sim,
            title=chunks.title,
            heading=chunks.heading,
            page_number=chunks.page
        )
        .limit(20)
    )

# IMAGE PROCESSING
# ===============
# Create a table for images - Pixeltable has first-class support for images
images = pxt.create_table(
    "agents.images",
    {"image": pxt.Image, "uuid": pxt.String, "timestamp": pxt.Timestamp, "user_id": pxt.String},
    if_exists="ignore",
)
print("Created/Loaded 'agents.images' table")

# Add computed column for image thumbnails (resized & encoded)
THUMB_SIZE_SIDEBAR = (96, 96)  # Define sidebar thumbnail size
images.add_computed_column(
    thumbnail=pxt_image.b64_encode(
        pxt_image.resize(images.image, size=THUMB_SIZE_SIDEBAR)
    ),
    if_exists="ignore",
)
print("Added/verified thumbnail computed column for images.")

# Insert local images from /data
print("Sample image insertion is disabled.") # Added notification

# Add an embedding index for images - This enables cross-modal search capabilities
# You can search images using text queries and vice versa
images.add_embedding_index(
    "image",
    embedding=clip.using(model_id=config.CLIP_MODEL_ID),  # Use config
    if_exists="ignore",
)


# Define an image search query - Similar syntax to document search
@pxt.query
def search_images(query_text: str, user_id: str):
    # Find similarity between the text embedding and image embeddings
    sim = images.image.similarity(query_text)  # Cross-modal similarity
    print(f"Image search query: {query_text} for user: {user_id}")
    # Return encoded images directly - Pixeltable handles image formats automatically
    return (
        images.where((images.user_id == user_id) & (sim > 0.25)) # Merged where clauses
        .order_by(sim, asc=False)
        .select(
            encoded_image=pxt_image.b64_encode(
                pxt_image.resize(images.image, size=(224, 224)), "png"
            ),  # Built-in image processing
            sim=sim,
        )
        .limit(5)
    )


# VIDEO PROCESSING
# ================
# Create a table for videos
videos = pxt.create_table(
    "agents.videos",
    {"video": pxt.Video, "uuid": pxt.String, "timestamp": pxt.Timestamp, "user_id": pxt.String},
    if_exists="ignore",
)
print("Created/Loaded 'agents.videos' table")

# Insert sample videos
print("Sample video insertion is disabled.") # Added notification

# Create Video Frames View
print("Creating video frames view...")
video_frames_view = pxt.create_view(
    "agents.video_frames",
    videos,
    iterator=FrameIterator.create(video=videos.video, fps=1),
    if_exists="ignore",
)
print("Created/Loaded 'agents.video_frames' view")

# Add Frame Embedding Index
print("Adding video frame embedding index (CLIP)...")
video_frames_view.add_embedding_index(
    column="frame",
    embedding=clip.using(model_id=config.CLIP_MODEL_ID),  # Use config
    if_exists="ignore",
)
print("Video frame embedding index created/verified.")


# Define Video Frame Search Query
@pxt.query
def search_video_frames(query_text: str, user_id: str):
    sim = video_frames_view.frame.similarity(query_text)
    print(f"Video Frame search query: {query_text} for user: {user_id}")
    return (
        video_frames_view.where((video_frames_view.user_id == user_id) & (sim > 0.25)) # Merged where clauses
        .order_by(sim, asc=False)
        .select(
            encoded_frame=pxt_image.b64_encode(video_frames_view.frame, "png"),
            source_video=video_frames_view.video,  # Include source video reference
            sim=sim,
        )
        .limit(5)  # Limit results
    )


# Extract audio from videos
videos.add_computed_column(
    audio=extract_audio(videos.video, format="mp3"), if_exists="ignore"
)

# AUDIO TRANSCRIPTION AND SEARCH
# ==============================
# Create view for video audio chunks
video_audio_chunks_view = pxt.create_view(
    "agents.video_audio_chunks",  # Renamed for clarity
    videos,
    iterator=AudioSplitter.create(
        audio=videos.audio,  # Use extracted audio
        chunk_duration_sec=30.0,  # Example: 30-second chunks
        overlap_sec=2.0,  # Example: 2-second overlap
    ),
    if_exists="ignore",
)

# Add transcription computed column to video audio chunks
print("Adding/Computing video audio transcriptions (OpenAI Whisper API)...")
video_audio_chunks_view.add_computed_column(
    transcription=openai.transcriptions(
        audio=video_audio_chunks_view.audio,
        model=config.WHISPER_MODEL_ID,  # Use config
    ),
    if_exists="replace",  # Keep replace for updates
)
print("Video audio transcriptions column added/updated.")

# Create view for video transcript sentences
video_transcript_sentences_view = pxt.create_view(
    "agents.video_transcript_sentences",  # Renamed for clarity
    video_audio_chunks_view.where(
        video_audio_chunks_view.transcription != None
    ),  # Ensure transcription exists
    iterator=StringSplitter.create(
        text=video_audio_chunks_view.transcription.text,  # Access the 'text' field
        separators="sentence",
    ),
    if_exists="ignore",
)

sentence_embed_model = sentence_transformer.using(
    model_id=config.EMBEDDING_MODEL_ID
)  # Use config

# Add embedding index for video transcript sentences
print("Adding video transcript sentence embedding index...")
video_transcript_sentences_view.add_embedding_index(
    column="text",
    string_embed=sentence_embed_model,
    if_exists="ignore",  # Ignore if already exists
)
print("Video transcript sentence embedding index created/verified.")


# Define video transcript search query
@pxt.query
def search_video_transcripts(query_text: str, user_id: str):
    sim = video_transcript_sentences_view.text.similarity(query_text)
    print(f"Video Transcript search query: {query_text} for user: {user_id}")
    return (
        video_transcript_sentences_view.where((video_transcript_sentences_view.user_id == user_id) & (sim > 0.8)) # Merged where clauses
        .order_by(sim, asc=False)
        .select(
            video_transcript_sentences_view.text,
            source_video=video_transcript_sentences_view.video,  # Select the source video from the view
            sim=sim,
        )
        .limit(5)
    )


# DIRECT AUDIO FILE PROCESSING
# ============================
# Create table for direct audio files
audios = pxt.create_table(
    "agents.audios",
    {"audio": pxt.Audio, "uuid": pxt.String, "timestamp": pxt.Timestamp, "user_id": pxt.String},
    if_exists="ignore",
)
print("Created/Loaded 'agents.audios' table")

# Insert sample audio files from /data
print("Sample audio insertion is disabled.") # Added notification

# Create view for direct audio chunks
audio_chunks_view = pxt.create_view(
    "agents.audio_chunks",  # New view for direct audio
    audios,
    iterator=AudioSplitter.create(
        audio=audios.audio,  # Use direct audio
        chunk_duration_sec=60.0,
        overlap_sec=2.0,
    ),
    if_exists="ignore",
)

# Add transcription computed column to direct audio chunks
print("Adding/Computing direct audio transcriptions (OpenAI Whisper API)...")
audio_chunks_view.add_computed_column(
    transcription=openai.transcriptions(
        audio=audio_chunks_view.audio,
        model=config.WHISPER_MODEL_ID,  # Use config
    ),
    if_exists="replace",  # Replace if definition changes
)
print("Direct audio transcriptions column added/updated.")

# Create view for direct audio transcript sentences
audio_transcript_sentences_view = pxt.create_view(
    "agents.audio_transcript_sentences",  # New view for direct audio sentences
    audio_chunks_view.where(audio_chunks_view.transcription != None),
    iterator=StringSplitter.create(
        text=audio_chunks_view.transcription.text, separators="sentence"
    ),
    if_exists="ignore",
)

# Add embedding index for direct audio transcript sentences
print("Adding direct audio transcript sentence embedding index...")
# Reuse the existing model definition
audio_transcript_sentences_view.add_embedding_index(
    column="text",
    string_embed=sentence_embed_model,
    if_exists="ignore",  # Ignore if already exists
)
print("Direct audio transcript sentence embedding index created/verified.")


# Define direct audio transcript search query
@pxt.query
def search_audio_transcripts(query_text: str, user_id: str):
    sim = audio_transcript_sentences_view.text.similarity(query_text)
    print(f"Direct Audio Transcript search query: {query_text} for user: {user_id}")
    return (
        audio_transcript_sentences_view.where((audio_transcript_sentences_view.user_id == user_id) & (sim > 0.8)) # Merged where clauses
        .order_by(sim, asc=False)
        .select(
            audio_transcript_sentences_view.text,
            source_audio=audio_transcript_sentences_view.audio,  # Select the source audio
            sim=sim,
        )
        .limit(5)
    )


# SELECTIVE MEMORY BANK (Code & Text)
# ==================================
# Create table for saved memory items
memory_bank = pxt.create_table(
    "agents.memory_bank",
    {
        "content": pxt.String,  # Code snippet or selected text
        "type": pxt.String,  # 'code' or 'text'
        "language": pxt.String,  # Programming language (for code type)
        "context_query": pxt.String,  # Query that generated the content
        "timestamp": pxt.Timestamp,
        "user_id": pxt.String
    },
    if_exists="ignore",
)

# Add embedding index for semantic search on content
print("Adding memory bank content embedding index...")
# Reuse the existing sentence transformer model
memory_bank.add_embedding_index(
    column="content",
    string_embed=sentence_embed_model,  # Defined earlier
    if_exists="ignore",
)
print("Memory bank content embedding index created/verified.")


# Define query to retrieve all memory items (for basic display)
@pxt.query
def get_all_memory(user_id: str):
    return memory_bank.where(memory_bank.user_id == user_id).select(
        content=memory_bank.content,
        type=memory_bank.type,
        language=memory_bank.language,
        context_query=memory_bank.context_query,
        timestamp=memory_bank.timestamp,
    ).order_by(memory_bank.timestamp, asc=False)


# Define query for semantic search on memory content
@pxt.query
def search_memory(query_text: str, user_id: str):
    sim = memory_bank.content.similarity(query_text)
    print(f"Memory Bank search query: {query_text} for user: {user_id}")
    return (
        memory_bank.where((memory_bank.user_id == user_id) & (sim > 0.8)) # Merged where clauses
        .order_by(sim, asc=False)
        .select(
            content=memory_bank.content,
            type=memory_bank.type,
            language=memory_bank.language,
            context_query=memory_bank.context_query,
            sim=sim,
        )
        .limit(10)  # Limit results
    )


# CHAT HISTORY TABLE & QUERY
# ================================
# Create table specifically for storing conversation turns
chat_history = pxt.create_table(
    "agents.chat_history",
    {
        "role": pxt.String,  # 'user' or 'assistant'
        "content": pxt.String,
        "timestamp": pxt.Timestamp,
        "user_id": pxt.String
    },
    if_exists="ignore",
)

# Add embedding index directly to chat history content
print("Adding chat history content embedding index...")
chat_history.add_embedding_index(
    column="content",
    string_embed=sentence_embed_model,  # Reuse existing model
    if_exists="ignore",
)
print("Chat history content embedding index created/verified.")


# Query to retrieve recent history from the dedicated table
@pxt.query
def get_recent_chat_history(user_id: str, limit: int = 4):  # Get last 4 messages (2 pairs)
    return (
        chat_history.where(chat_history.user_id == user_id)
        .order_by(chat_history.timestamp, asc=False)
        .select(role=chat_history.role, content=chat_history.content)
        .limit(limit)
    )


# Query for semantic search on the entire chat history
@pxt.query
def search_chat_history(query_text: str, user_id: str):
    sim = chat_history.content.similarity(query_text)
    print(f"Chat History search query: {query_text} for user: {user_id}")
    return (
        chat_history.where((chat_history.user_id == user_id) & (sim > 0.8)) # Merged where clauses
        .order_by(sim, asc=False)
        .select(role=chat_history.role, content=chat_history.content, sim=sim)
        .limit(10)  # Limit results
    )


# USER PERSONAS TABLE
# =====================
# Create table to store user-defined personas (prompts + params)
user_personas = pxt.create_table(
    "agents.user_personas",
    {
        "user_id": pxt.String,
        "persona_name": pxt.String,
        "initial_prompt": pxt.String,
        "final_prompt": pxt.String,
        "llm_params": pxt.Json,
        "timestamp": pxt.Timestamp,
    },
    if_exists="ignore",
)
print("Created/Loaded 'agents.user_personas' table")


# IMAGE GENERATION PIPELINE
# ===============================
# Create table for image generation requests
image_gen_tasks = pxt.create_table(
    "agents.image_generation_tasks",
    {"prompt": pxt.String, "timestamp": pxt.Timestamp, "user_id": pxt.String},
    if_exists="ignore",
)

# Add computed column to generate image using OpenAI
image_gen_tasks.add_computed_column(
    generated_image=openai.image_generations(
        prompt=image_gen_tasks.prompt,
        model=config.DALLE_MODEL_ID,  # Use config
        size="1024x1024",  # Specify a default size
        # Add other parameters like quality, style if desired
    ),
    if_exists="ignore",
)
print("Image generation table and computed column created/verified.")

# AGENT WORKFLOW DEFINITION
# ========================
# Register tools for LLM use - Pixeltable has built-in LLM tool integration
tools = pxt.tools(
    functions.get_latest_news,
    functions.fetch_financial_data,
    functions.search_news
)

# Create the main workflow table - Pixeltable excels at orchestrating complex workflows
tool_agent = pxt.create_table(
    "agents.tools",
    {
        "prompt": pxt.String,
        "timestamp": pxt.Timestamp,
        "user_id": pxt.String,
        "initial_system_prompt": pxt.String,
        "final_system_prompt": pxt.String,
        "max_tokens": pxt.Int,
        "stop_sequences": pxt.Json,
        "temperature": pxt.Float,
        "top_k": pxt.Int,
        "top_p": pxt.Float,
    },
    if_exists="ignore",
)

# DECLARATIVE WORKFLOW WITH COMPUTED COLUMNS
# =========================================
# This is where Pixeltable shines - define your entire AI processing pipeline declaratively
# Computed columns automatically execute when their dependencies change

# Initial LLM reasoning - Pixeltable has built-in Anthropic integration
tool_agent.add_computed_column(  # Add a column that's computed from other columns
    initial_response=messages(  # Call the Anthropic Claude API
        model=config.CLAUDE_MODEL_ID,  # Use config
        system=tool_agent.initial_system_prompt,
        messages=[{"role": "user", "content": tool_agent.prompt}],
        tools=tools,  # Pass tools for function calling,
        tool_choice=tools.choice(required=True),
        # Pass new parameters
        max_tokens=tool_agent.max_tokens,
        stop_sequences=tool_agent.stop_sequences,
        temperature=tool_agent.temperature,
        top_k=tool_agent.top_k,
        top_p=tool_agent.top_p,
    ),
    if_exists="replace",
)

# Tool execution - This column automatically invokes the tools identified by Claude
tool_agent.add_computed_column(
    tool_output=invoke_tools(tools, tool_agent.initial_response), if_exists="replace"
)

# Document search - Pixeltable automatically optimizes the execution plan
tool_agent.add_computed_column(
    doc_context=search_documents(
        tool_agent.prompt, tool_agent.user_id
    ),  # Reusing the query function defined earlier
    if_exists="replace",
)

# Image search - Pixeltable's DAG-based execution engine handles dependencies
tool_agent.add_computed_column(
    image_context=search_images(tool_agent.prompt, tool_agent.user_id), if_exists="replace"
)

# Add Video Transcript Search Context (Existing, renamed variable)
tool_agent.add_computed_column(
    video_transcript_context=search_video_transcripts(
        tool_agent.prompt, tool_agent.user_id
    ),  # Use renamed query
    if_exists="ignore",
)

# Add *Direct* Audio Transcript Search Context
tool_agent.add_computed_column(
    audio_transcript_context=search_audio_transcripts(
        tool_agent.prompt, tool_agent.user_id
    ),  # Use new query
    if_exists="ignore",
)

# Add Video Frame Search Context
tool_agent.add_computed_column(
    video_frame_context=search_video_frames(tool_agent.prompt, tool_agent.user_id), if_exists="ignore"
)

# INTEGRATE MEMORY BANK SEARCH INTO AGENT WORKFLOW
# ================================================
# Add Memory Bank Search Context
tool_agent.add_computed_column(
    memory_context=search_memory(tool_agent.prompt, tool_agent.user_id), if_exists="ignore"
)

# Add Chat History Semantic Search Context
tool_agent.add_computed_column(
    chat_memory_context=search_chat_history(tool_agent.prompt, tool_agent.user_id), if_exists="ignore"
)

# Add Chat History Context (Now querying agents.chat_history)
tool_agent.add_computed_column(
    history_context=get_recent_chat_history(tool_agent.user_id),
    if_exists="ignore",  # Ignore if already exists from previous attempt
)

# Assemble the multimodal context string (excluding history)
tool_agent.add_computed_column(
    multimodal_context_summary=functions._assemble_multimodal_context(
        tool_agent.prompt,
        tool_agent.tool_output,
        tool_agent.doc_context,
        tool_agent.video_transcript_context,
        tool_agent.audio_transcript_context,
        tool_agent.memory_context,
        tool_agent.chat_memory_context,  # Pass the new context
    ),
    if_exists="replace",  # Replace if function signature changed
)

# Assemble the final message list for the LLM
tool_agent.add_computed_column(
    final_prompt_messages=functions.assemble_final_messages(
        tool_agent.history_context,
        tool_agent.multimodal_context_summary,
        image_context=tool_agent.image_context,  # Pass image context separately
        video_frame_context=tool_agent.video_frame_context,  # Pass video frame context
    ),
    if_exists="replace",  # Replace if function signature changed
)

# Final LLM reasoning - Now uses the pre-assembled message list
tool_agent.add_computed_column(
    final_response=messages(
        model=config.CLAUDE_MODEL_ID,  # Use config
        system=tool_agent.final_system_prompt,
        messages=tool_agent.final_prompt_messages,  # Use the new computed column
        max_tokens=tool_agent.max_tokens,
        stop_sequences=tool_agent.stop_sequences,
        temperature=tool_agent.temperature,
        top_k=tool_agent.top_k,
        top_p=tool_agent.top_p,
    ),
    if_exists="replace",
)

# Answer extraction - Simple transformations are also computed columns
tool_agent.add_computed_column(
    # Extract the primary text content from the structured LLM response
    answer=tool_agent.final_response.content[0].text,
    if_exists="replace",
)

# NEW: Add computed column to construct the input message for Mistral
tool_agent.add_computed_column(
    follow_up_input_message=functions.assemble_follow_up_prompt(
        original_prompt=tool_agent.prompt, answer_text=tool_agent.answer
    ),
    if_exists="replace",
)

# Generate the raw response from Mistral first for easier debugging
tool_agent.add_computed_column(
    follow_up_raw_response=mistral_chat(
        model=config.MISTRAL_MODEL_ID,  # Use config
        messages=[
            {
                "role": "user",
                "content": tool_agent.follow_up_input_message,
            }  # Use the new computed column
        ],
        max_tokens=150,  # Limit output size
        temperature=0.6,  # Slightly creative but mostly grounded
    ),
    if_exists="replace",  # Use replace during development/refinement
)

# Extract the text content from the raw response
tool_agent.add_computed_column(
    follow_up_text=tool_agent.follow_up_raw_response.choices[
        0
    ].message.content,  # Extract string from the raw response column
    if_exists="replace",  # Use replace during development/refinement
)
