def get_event_data():
    '''
    Retrieve current cultural events occuring in Seoul.

    Return:
        Dataset containing the event datas.
    '''
    return [] # we only need the name of function to identify when to use RAG.

def display_path(target_location: str):
    '''
    Display the path from user location to the target location utilizing external APIs.
    User locations are handled from front-end.

    Args:
        target_location: Name of the target location e.g. 청량리역, 강남역 스타벅스 etc.
    
    Returns:
        A url containing the path.
    '''
    return 'https://naver.com'
