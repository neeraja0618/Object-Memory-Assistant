"""Search tentative evidence without promoting it to a confirmed identity."""
def possible_sightings(rows, query, excluded=None, rejected=None):
    query=query.strip().casefold()
    if not query:
        return []
    excluded=excluded or {}
    rejected=rejected or set()
    matches=[]
    for row in rows:
        if row.get('state')!='Tentative' or row['track'] in rejected:
            continue
        if row.get('candidate_id') in excluded.get(row['track'],set()):
            continue
        candidate=row.get('candidate_name','').casefold()
        detector=row.get('detector_name','').casefold()
        if query in candidate or query==detector:
            matches.append(row)
    return sorted(matches,key=lambda row:row['seconds'],reverse=True)
