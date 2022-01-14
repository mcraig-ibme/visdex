def main():
    # Default logging to stdout
    import logging
    logging.basicConfig(level=logging.INFO)

    from visdex.app import app
    app.run_server(debug=False)

main()
