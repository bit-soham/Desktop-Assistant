import urllib.parse, socket
url = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
host = urllib.parse.urlparse(url).hostname
print("host:", host)
try:
    print("getaddrinfo:", socket.getaddrinfo(host, 443))
except Exception as e:
    print("DNS/resolve error:", repr(e))
