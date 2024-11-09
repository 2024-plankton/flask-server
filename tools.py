def get_event_data():
    '''
    Retrieve current cultural events occuring in Seoul.

    Return:
        Dataset containing the event datas.
    '''
    return [] # we only need the name of function to identify when to use RAG.

def get_unusual_activity():
    '''
    Retrieve list of fun and unusual activities for daily fun!

    Return:
        List of fun and unusual activities, just for you.
    '''
    return [] # as we have to hard-code these...

def search_youtube_video(query: str):
    '''
    Return youtube video link.

    Args:
        query: search query for youtube.
    
    Return:
        youtube video link.
    '''
    return '' # as we only need query string...


def display_map(target_location: str):
    '''
    Display the map for the target location utilizing external APIs.
    User locations are handled from front-end.

    Args:
        target_location: Name of the target location e.g. 청량리역, 강남역 스타벅스 etc.
    
    Returns:
        A url containing the path.
    '''
    return 'https://naver.com'
