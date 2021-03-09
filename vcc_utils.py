import re
import psycopg2
import psycopg2.extras

DBNAME = "vccfinder"
USER = None
HOST = "/var/run/postgresql"  # hostname, IP or unix socket dir
PASSWORD = None

PG = psycopg2.connect(f"dbname={DBNAME} host={HOST}")
# We have to let psycopg2 that hstore should be casted to dict (not enabled by default)
psycopg2.extras.register_hstore(PG, globally=True)


def get_commit_from_db(commit_id):
    dict_cur = PG.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    dict_cur.execute("SELECT id, message, patch FROM export.commits WHERE id=%s", (commit_id, ))
    res = dict_cur.fetchone()
    if not res:
        return None
    #treated_res = filter_commit(res)
    dict_cur.close()
    assert('id' in res)
    assert('message' in res)
    assert('patch' in res)
    tmp = dict()
    tmp['id'] = res['id']
    tmp['message'] = res['message']
    tmp['patch'] = res['patch'].split('\n')
    return tmp

def get_commits_from_db(commit_ids):
    """
    Returns a generator of (commit_id, commit_message, commit_patch)
    """
    for commit_id in commit_ids:
        commit = get_commit_from_db(commit_id)
        # We are not supposed to know about commits that are not in the database
        assert(commit is not None)
        yield (commit['id'], commit['message'], commit['patch'])


CODE_FILE_EXTENSIONS = ['.c', '.cpp', '.cc', '.h', '.java']


def is_code_file(extension_list):
    for i in extension_list:
        if i in CODE_FILE_EXTENSIONS:
            return True
    return False



EXTENSIONS_FINDER = re.compile('(\.\w+)')
def filter_non_code(commit_flow):
    """
    Takes a generator of (commit_id, commit_message, commit_patch) as provided by get_commits()
    Returns a generator of (commit_id, commit_txt) where diffs of non-code files are removed
    """
    for (commit_id, commit_message, commit_patch) in commit_flow:
        new_commit_patch = list()
        keep_skipping = False
        for line in commit_patch:
            if ("diff --git" in line):
                    # So... this is a change to a file
                    # collect everything that looks like a file extension
                    matches = EXTENSIONS_FINDER.findall(line)
                    # Does this line mention a filetype we are interested in ?
                    if is_code_file(matches):
                        keep_skipping = False
                        new_commit_patch.append(line)
                        continue
                    else:
                        keep_skipping = True
                        continue
            if keep_skipping:
                continue
            else:
                new_commit_patch.append(line)
                continue
        yield (commit_id, commit_message, new_commit_patch)
