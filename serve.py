#!/usr/bin/env python

from http.server import BaseHTTPRequestHandler, HTTPServer
from hmm_pos import pos_tagger
from urllib.parse import urlparse
from urllib import parse

class HTTPServerRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        if self.path.startswith("/pos_tagger?sent="):
            url = urlparse(self.path)
            res = pos_tagger(parse.unquote(url.query[5:]))
            message = ", ".join(res[0])
            self.wfile.write(bytes(message, "utf8"))
        elif self.path == "/":
            self.wfile.write(open("slides.html", "rb").read())
        return


def run():
    print('starting server...')
    server_address = ('127.0.0.1', 8080)
    httpd = HTTPServer(server_address, HTTPServerRequestHandler)
    print('running server...')
    httpd.serve_forever()


run()