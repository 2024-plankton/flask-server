def get_event_data():
    '''
    현재 서울에서 진행중인 행사/공연 등의 다양한 정보를 가져옵니다.

    Return:
        Dataset containing the event datas.
    '''
    return [] # we only need the name of function to identify when to use RAG.

def search_youtube_video(query: str):
    '''
    관련있는 유튜브 비디오 링크를 가져옵니다.

    Args:
        query: search query for youtube.
    
    Return:
        youtube video link.
    '''
    return '' # as we only need query string...


def display_map(target_location: str):
    '''
    지도 API를 활용하여 목적지에 대한 정보를 받아옵니다.
    출발지의 정보는 front-end에서 처리됩니다.

    Args:
        target_location: Name of the target location e.g. 청량리역, 강남역 스타벅스 etc.
    
    Returns:
        A url containing the path.
    '''
    return 'https://naver.com'
