from contract_env.server.app import app


def main():
    import uvicorn
    uvicorn.run(
        "contract_env.server.app:app",
        host="0.0.0.0",
        port=7860,
    )


if __name__ == "__main__":
    main()