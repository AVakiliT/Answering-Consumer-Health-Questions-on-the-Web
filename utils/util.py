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

def unfixdocno(s):
    return f"c4-{int(s[21:25]):04}-{int(s.split('.')[-1]):06}"


import subprocess

topics_2019 = list(range(1, 51 + 1))
topics_2021 = list(range(101, 150 + 1))
topics_2022 = list(range(151, 200 + 1))
topics_rw = list(range(1001, 1090 + 1))

def shell_cmd(command):
    """Executes the given command within terminal and returns the output as a string

    :param command: the command that will be executed in the shell
    :type command: str

    :return: the output of the command
    :rtype: str
    """
    process = subprocess.run([command], stderr=subprocess.PIPE, shell=True, universal_newlines=True)

    return process