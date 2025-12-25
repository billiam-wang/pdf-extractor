# Databricks Model Serving Setup

This guide explains how to configure the app to use Databricks Model Serving endpoints instead of OpenAI/Anthropic API keys.

## Configuration Options

### Option 1: Full Endpoint URL (Simplest)

Set the complete serving endpoint URL:

```bash
export LLM_PROVIDER=databricks
export DATABRICKS_SERVING_ENDPOINT=https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint-name/invocations
```

For external calls (not in Databricks Apps), also add:
```bash
export DATABRICKS_TOKEN=your-databricks-personal-access-token
```

### Option 2: Host + Endpoint Name

Alternatively, configure using separate variables:

```bash
export LLM_PROVIDER=databricks
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_ENDPOINT_NAME=your-endpoint-name
export DATABRICKS_TOKEN=your-token  # Only for external calls
```

## Databricks Apps (Automatic Auth)

When running as a **Databricks App**, authentication is automatic:

1. Set environment variables in your app configuration:
   ```yaml
   env:
     - name: LLM_PROVIDER
       value: "databricks"
     - name: DATABRICKS_SERVING_ENDPOINT
       value: "https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint-name/invocations"
   ```

2. **No token needed** - Databricks Apps automatically authenticate with workspace identity

## Finding Your Endpoint URL

1. Go to your Databricks workspace
2. Navigate to **Serving** in the left sidebar
3. Click on your model serving endpoint
4. Copy the **Invocation URL** (looks like: `https://....cloud.databricks.com/serving-endpoints/your-endpoint-name/invocations`)

## Supported Model Formats

The app handles multiple response formats automatically:

### OpenAI-Compatible Format
```json
{
  "choices": [{
    "message": {
      "content": "extracted specifications..."
    }
  }]
}
```

### Databricks MLflow Format
```json
{
  "predictions": ["extracted specifications..."]
}
```

### Direct Content Format
```json
{
  "content": "extracted specifications..."
}
```

## Request Format

The app sends requests in this format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a technical document analyzer..."
    },
    {
      "role": "user",
      "content": "Analyze the following technical document..."
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.1
}
```

**Note:** Adjust the request format in `parser/technical_drawing_parser.py` if your model expects a different structure.

## Testing Locally

To test with a Databricks endpoint locally:

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your configuration
# LLM_PROVIDER=databricks
# DATABRICKS_SERVING_ENDPOINT=https://...
# DATABRICKS_TOKEN=dapi...

# Load environment variables
source .env  # or use direnv, dotenv, etc.

# Run the app
python app.py
```

## Troubleshooting

### "Error: DATABRICKS_HOST or DATABRICKS_SERVING_ENDPOINT must be set"

Make sure you've set one of:
- `DATABRICKS_SERVING_ENDPOINT` (full URL), or
- Both `DATABRICKS_HOST` and `DATABRICKS_ENDPOINT_NAME`

### "Error: DATABRICKS_ENDPOINT_NAME not set"

If using Option 2, you must set `DATABRICKS_ENDPOINT_NAME` along with `DATABRICKS_HOST`.

### Authentication Errors (401/403)

**In Databricks Apps:**
- Authentication should be automatic
- Ensure your app has permission to access the serving endpoint

**External Calls:**
- Set `DATABRICKS_TOKEN` with a valid personal access token
- Create token at: Workspace Settings → Developer → Access Tokens

### Timeout Errors

The default timeout is 60 seconds. For large documents, you might need to:
1. Increase the timeout in `_call_databricks_endpoint()` method
2. Optimize your model's response time
3. Process documents in smaller chunks

## Model Requirements

Your Databricks serving endpoint should:

1. Accept chat-formatted messages (system + user roles)
2. Support `max_tokens` parameter
3. Return responses in one of the supported formats above
4. Handle technical document analysis prompts

## Example: Deploy Foundation Model

If you're using a Databricks Foundation Model (e.g., Meta Llama, Mistral):

1. Create serving endpoint in Databricks UI
2. Select your foundation model
3. Configure endpoint name
4. Copy the invocation URL
5. Use in this app

That's it! The app handles the rest automatically.
