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

def fixdocno(x):
    return f"en.noclean.c4-train.0{x[3:7]}-of-07168.{int(x[8:])}"