"""Public-data download and a clearly labelled ORBIT-style component-test manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from zipfile import ZipFile

SOURCE='https://zenodo.org/records/12608444'
URL='https://zenodo.org/records/12608444/files/ORBIT-India%20dataset.zip?download=1'
CHECKSUM='af4d05689f556ca477acc09a73bb3d3d'
EXTENSIONS={'.jpg','.jpeg','.png'}


def download(destination):
    destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'orbit_india.zip'
    temporary=archive.with_suffix('.part')
    if not archive.exists():
        request=urllib.request.Request(URL,headers={'User-Agent':'PersonalObjectMemoryResearch/1.0'})
        print('Downloading the public ORBIT-India archive (~1.3 GB). Source:',SOURCE,flush=True)
        with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as output:
            total=0;announced=0
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                output.write(chunk);total+=len(chunk)
                if total-announced>=50*1024*1024:print(f'{total/1024**2:.0f} MiB downloaded',flush=True);announced=total
        temporary.replace(archive)
    digest=hashlib.md5()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=CHECKSUM:raise ValueError('Checksum mismatch. Delete the incomplete/changed archive and check the source record.')
    extraction=destination/'orbit_india'
    extraction.mkdir(exist_ok=True)
    with ZipFile(archive) as z:
        for member in z.infolist():
            target=(extraction/member.filename).resolve()
            if not target.is_relative_to(extraction.resolve()):raise ValueError('Unsafe archive member path.')
            if member.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as source,target.open('wb') as output:shutil.copyfileobj(source,output)
    print('Extracted to:',extraction)
    return extraction


def build_manifest(root,output,collector=None,max_objects=6,max_clean=30,max_query=30):
    root=root.resolve()
    annotations={}
    for path in root.rglob('*.json'):
        if path.parent.name.lower() not in ('annotations','annotation'):continue
        try:
            data=json.loads(path.read_text())
            if isinstance(data,dict):annotations.update(data)
        except (ValueError,UnicodeDecodeError):pass
    candidates={}
    for mode_folder in root.rglob('*'):
        if not mode_folder.is_dir() or mode_folder.name.lower() not in ('clean','clutter'):continue
        identity=mode_folder.parent.name
        user=mode_folder.parent.parent.name
        if collector and user!=collector:continue
        candidates.setdefault(user,{}).setdefault(identity,{'clean':[],'clutter':[]})[mode_folder.name.lower()].extend(
            p for p in mode_folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    usable={user:{name:modes for name,modes in objects.items() if modes['clean'] and modes['clutter']} for user,objects in candidates.items()}
    usable={user:objects for user,objects in usable.items() if len(objects)>=2}
    if not usable:raise ValueError('Need at least two objects for one collector, each with clean and clutter image folders. Check extraction/layout.')
    user=collector if collector in usable else sorted(usable,key=lambda u:(-len(usable[u]),u))[0]
    objects=[]
    def sample(paths,limit):
        paths=sorted(set(paths))
        paths=[p for p in paths if not annotations.get(p.name,{}).get('pii_present_issue',False)]
        if len(paths)>limit:
            indices=[round(i*(len(paths)-1)/(limit-1)) for i in range(limit)] if limit>1 else [0]
            paths=[paths[i] for i in indices]
        return paths
    for name,modes in sorted(usable[user].items())[:max_objects]:
        clean=[p for p in sample(modes['clean'],max_clean) if not annotations.get(p.name,{}).get('object_not_present_issue',False)]
        query=sample(modes['clutter'],max_query)
        if not clean or not query:continue
        objects.append(dict(id=name,name=name,references=[str(p) for p in clean],queries=[dict(path=str(p),
            object_present=not annotations.get(p.name,{}).get('object_not_present_issue',False),
            presence_annotation_available=p.name in annotations) for p in query]))
    if len(objects)<2:raise ValueError('Too few valid objects after filtering. Choose another collector or inspect annotations.')
    manifest=dict(source=SOURCE,collector=user,protocol='Custom clean/clutter appearance-component diagnostic; not the official ORBIT benchmark or a chronological multi-object memory test',objects=objects)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Manifest saved:',output,'| objects:',len(objects))
    return manifest


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    fetch=sub.add_parser('download');fetch.add_argument('--destination',type=Path,default=Path('datasets'))
    prepare=sub.add_parser('manifest');prepare.add_argument('root',type=Path);prepare.add_argument('--output',type=Path,default=Path('evaluation/public_manifest.json'))
    prepare.add_argument('--collector');prepare.add_argument('--max-objects',type=int,default=6)
    args=parser.parse_args()
    if args.command=='download':download(args.destination)
    else:build_manifest(args.root,args.output,args.collector,args.max_objects)


if __name__=='__main__':main()
