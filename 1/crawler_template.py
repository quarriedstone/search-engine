from html.parser import HTMLParser
from urllib.request import urlopen
from urllib import parse


# This class inherits a method from HTMLParser and adds a new one
class LinkParser(HTMLParser):
    # This is a function that HTMLParser normally has
    # but we are adding some functionality to it

    def handle_starttag(self, tag, attrs):
        # We are looking for the beginning of a link. Links normally look
        # like <a href="www.someurl.com"></a>
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    # We are grabbing the new URL. We are also adding the
                    # base URL to it. For example:
                    # www.netinstructions.com is the base and
                    # somepage.html is the new URL (a relative URL)
                    #
                    # We combine a relative URL with the base URL to create
                    # an absolute URL like:
                    # www.netinstructions.com/somepage.html
                    newUrl = parse.urljoin(self.baseUrl, value)
                    # And add it to our collection of links:
                    self.links = self.links + [newUrl]

    # This is a new function that we are creating to get links
    # that our spider() function will call
    def getLinks(self, url):
        self.links = []
        # Remember the base URL which will be important when creating
        # absolute URLs
        self.baseUrl = url
        # Use the urlopen function from the standard Python 3 library
        response = urlopen(url)
        # Make sure that we are looking at HTML and not other things that
        # are floating around on the internet (such as
        # JavaScript files, CSS, or .PDFs for example)
        if 'text/html' in response.getheader('Content-Type'):
            htmlBytes = response.read()
            # Note that feed() handles Strings well, but not bytes
            # (A change from Python 2.x to Python 3.x)
            htmlString = htmlBytes.decode("utf-8")
            self.feed(htmlString)
            return htmlString, self.links
        else:
            return "", []


# And finally here is our spider. It takes in an URL, a word to find,
# and the number of pages to search through before giving up
def spider(url, word, maxPages):
    parser = LinkParser()
    parser.handle_starttag(word, parser.getLinks(url))

    # Saving only unique urls
    curr_links = []
    ban = set()
    for link in parser.links:
        if link not in ban:
            curr_links.append(link)
        ban.add(link)

    for i in range(1, maxPages + 1):
        if len(curr_links) == 0:
            break

        link = curr_links.pop(0)
        try:
            tag = parser.getLinks(link)
            count = len(parser.links)
            print(str(i), "Visiting: ", link, "Have got", count, "links")

            if (tag[0]).find(word) != -1:
                print("**Word found!**")

            for l in tag[1]:
                if l not in ban:
                    curr_links.append(l)
                ban.add(l)
        except:
            print("ERROR")

spider("http://www.innopolis.com/", "infrastructure", 100)
