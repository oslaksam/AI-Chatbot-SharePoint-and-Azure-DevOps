import re

class DocumentFilter:
    def apply(self, doc):
        """
        Apply the filter to the document.

        Parameters:
        - doc: A document object.

        Returns:
        - bool: True if the document passes the filter, False otherwise.
        """
        raise NotImplementedError("This method needs to be implemented by subclasses.")


class CompositeFilter(DocumentFilter):
    def __init__(self, filters=None):
        self.filters = filters if filters is not None else []

    def add_filter(self, filter):
        self.filters.append(filter)

    def apply(self, doc):
        return all(filter.apply(doc) for filter in self.filters)


class HomepageFilter(DocumentFilter):
    def __init__(self):
        self.pattern = re.compile(r"homepage", re.IGNORECASE)

    def apply(self, doc):
        regex = bool(self.pattern.search(doc.page_content))
        isHomeLayout = doc.metadata.get("pageLayout") == "home"
        isNoDescription = doc.metadata.get("description") == "No Description"
        return not bool(isHomeLayout)

class NewsFilter(DocumentFilter):
    def __init__(self):
        self.pattern = re.compile(r"TEMPLATE in the news", re.IGNORECASE)

    def apply(self, doc):
        isNoDescription = doc.metadata.get("description") == "No Description"
        isNews = isNoDescription and bool(self.pattern.search(doc.page_content))
        return not isNews

class LocationsFilter(DocumentFilter):
    def __init__(self):
        self.pattern = re.compile(r"TEMPLATE Locations", re.IGNORECASE)

    def apply(self, doc):
        isNoDescription = doc.metadata.get("description") == "No Description"
        isLocations = isNoDescription and bool(self.pattern.search(doc.page_content))
        return not isLocations

