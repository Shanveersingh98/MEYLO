import socket

for host in ["duckduckgo.com", "html.duckduckgo.com", "lite.duckduckgo.com", "api.duckduckgo.com"]:
    try:
        ips = socket.getaddrinfo(host, 443)
        print(f"Host: {host} -> {[x[4][0] for x in ips[:2]]}")
    except Exception as e:
        print(f"Host: {host} DNS error: {e}")
