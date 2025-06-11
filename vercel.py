{
    "version": 2,
    "builds": [
        {
            "src": "main.py",
            "use": "@vercel/python"
        }
    ],
    "routes": [
        {
            "src": "/(.*)",
            "dest": "main.py"
        }
    ],
    "env": {
        "AIPROXY_TOKEN": "@aiproxy_token",
        "PINECONE_API_KEY": "@pinecone_api_key"
    }
}
