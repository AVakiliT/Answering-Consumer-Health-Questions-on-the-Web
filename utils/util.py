from tldextract import extract


def url2host(url):
    e = extract(url)
    if e.domain != "www" or e.domain == "":
        return f"{(e.subdomain + '.') if e.subdomain and e.subdomain != 'www' else ''}{e.domain}.{e.suffix}"
    else:
        return e.suffix

def url2domain(url):
    e = extract(url)
    return f"{e.domain}.{e.suffix}"