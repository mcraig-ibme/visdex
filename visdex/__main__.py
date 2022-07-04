def main():
    # Default logging to stdout
    import logging
    logging.basicConfig(level=logging.INFO)

    from visdex.app import dash_app
    dash_app.run_server(debug=False)

main()
