import pandas as pd

from itertools import chain
def playlist_dicts(*files):
    """
        NOTE: all playlists
        given arbitrary many filename(s) (exact filename... must include directory),
        returns a dict of uid's and 1's, ready for dataframe for matrix factorization
    """
    playlists = list(chain(*[pd.read_json(x, typ='series')['playlists'] for x in files]))
    users = [({x['track_uri'].split(':')[-1]:1 for x in l['tracks']}) for l in playlists]
    return users

def read_playlist(filename, index):
    """
        NOTE: single playlist
        given a filename (exact filename... must include directory),
        returns a tuple of playlist details at index, and playlist tracks
    """
    playlist_df = pd.read_json(filename, typ='series')
    
    # this line is kind of a disaster, there might be a cleaner way
    current_details = pd.DataFrame(playlist_df['playlists'][index]).drop('tracks', axis=1).iloc[0]
    current_tracks = pd.DataFrame(playlist_df['playlists'][index]['tracks'])
    
    return current_details, current_tracks



def read_playlists(filename):
    """
        NOTE: all playlists
        given a filename (exact filename... must include directory),
        returns a list of tuple of playlist details, and playlist tracks, for all playlists
    """
    playlist_df = pd.read_json(filename, typ='series')
    pl_lst = []
    
    for pl in playlist_df['playlists']:
        # this line is kind of a disaster, there might be a cleaner way
        current_details = pd.DataFrame(pl).drop('tracks', axis=1).iloc[0]
        current_tracks = pd.DataFrame(pl['tracks'])
        pl_lst.append((current_details, current_tracks))
    
    return pl_lst

def read_playlists_uri(filename):
    """
        NOTE: all playlists
        given a filename (exact filename... must include directory),
        returns a list of track uri for each playlist
    """
    playlist_df = pd.read_json(filename, typ='series')
    pl_lst = []

    for pl in playlist_df['playlists']:
        current_tracks = pd.DataFrame(pl['tracks'])
        current_tracks = current_tracks.apply(lambda x : x['track_uri'].split(':')[-1], axis=1)
        
        pl_lst.append(pd.DataFrame(current_tracks, columns=['track_uri']))

    return pl_lst

def get_metadata(playlist_df, metadata_df):
    """
        NOTE: for a single playlist
        given a playlist dataframe (current_tracks from read_playlist), and a dataframe of song
        metadata (must read in from '/workspaces/codespaces-jupyter/data/song_metadata/data.csv'
        before calling get_metadata)
        returns dataframe including metadata for tracks in playlist
    """
    # remove 'spotify:track:' crap
    playlist_df['track_uri'] = playlist_df.apply(lambda x : x['track_uri'].split(':')[-1], axis=1)
    playlist_df['artist_uri'] = playlist_df.apply(lambda x : x['artist_uri'].split(':')[-1], axis=1)
    playlist_df['album_uri'] = playlist_df.apply(lambda x : x['album_uri'].split(':')[-1], axis=1)
    
    # join with metadata dataframe on id number
    df = pd.merge(playlist_df.drop('duration_ms', axis=1),
                  metadata_df, left_on='track_uri',
                  right_on='id',
                  how='inner')
    
    # EXPERIMENTAL: DROP EXPLICIT, POS (not sure they contribute)
    return df.drop(['explicit', 'pos'], axis=1)