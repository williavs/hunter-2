import os
import openai
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

# Load API key from .env file, ignoring environment variables
config = dotenv_values(".env") # Assuming .env is in the parent directory
api_key = config.get("API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not found in .env file.")
    exit()

# --- Configuration ---
custom_base_url = "https://llm.data-qa.justworks.com" # Corrected base URL (no leading space)
model_to_test = "openai/gpt-4.1" # Use the model specified in your analyzer
test_prompt = "What is the weather in New York today?"
# --- End Configuration ---

print(f"Attempting to call Responses API at: {custom_base_url}")
print(f"Using model: {model_to_test}")
print(f"Using API key: {api_key[:5]}...{api_key[-4:]}")

try:
    # Initialize OpenAI client pointing to the custom base URL
    client = openai.OpenAI(
        api_key=api_key,
        base_url=custom_base_url,
    )

    # Make the call using the /responses endpoint structure
    print(f"\nSending request with prompt: '{test_prompt}' and web_search_preview tool...")
    response = client.responses.create(
        model=model_to_test,
        tools=[{"type": "web_search_preview"}],
        input=test_prompt
    )

    print("\n--- Success! ---")
    print("Response received:")
    # Pretty print the response object if possible, otherwise just print it
    try:
        import json
        print(json.dumps(response.model_dump(), indent=2))
    except:
        print(response)

except openai.NotFoundError as e:
    print(f"\n--- API Error (Not Found) ---")
    print(f"The endpoint might be incorrect or not implemented on the server.")
    print(f"Status Code: {e.status_code}")
    print(f"Error details: {e}")

except openai.APIConnectionError as e:
    print(f"\n--- API Error (Connection) ---")
    print("Could not connect to the server. Check the base URL and network connectivity.")
    print(f"Error details: {e}")
    
except openai.AuthenticationError as e:
    print(f"\n--- API Error (Authentication) ---")
    print("Authentication failed. Check if the API key is valid for this endpoint.")
    print(f"Status Code: {e.status_code}")
    print(f"Error details: {e}")

except openai.RateLimitError as e:
    print(f"\n--- API Error (Rate Limit) ---")
    print("Rate limit exceeded.")
    print(f"Status Code: {e.status_code}")
    print(f"Error details: {e}")

except openai.APIError as e:
    print(f"\n--- General API Error ---")
    print(f"An unexpected API error occurred.")
    print(f"Status Code: {e.status_code}")
    print(f"Error details: {e}")

except Exception as e:
    print(f"\n--- Unexpected Error ---")
    print(f"An error occurred: {e}")

# --- LangChain Configuration ---
llm = ChatOpenAI(
    api_key=api_key,
    model="openai/gpt-4.1",  # Use the model name that works for your endpoint
    base_url="https://llm.data-qa.justworks.com",  # <--- Correct!
    use_responses_api=True,
    timeout=60,
)

tool = {"type": "web_search_preview"}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke("What is the weather in New York today?")
print(response.text()) 