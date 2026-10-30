"""Optional seed references from a downloaded public manifest, without pretending they are the user's objects."""
import argparse
import json
from pathlib import Path
from PIL import Image
from personal import Registry


def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('--views',type=int,default=5)
    args=parser.parse_args();data=json.loads(args.manifest.read_text());root=Path(__file__).resolve().parent
    registry=Registry(root/'data');mapping={}
    for item in data['objects']:
        name='Public demo: '+item['name']
        existing=next((o for o in registry.active() if o['name']==name),None)
        registration=existing or registry.create(name);mapping[item['id']]=registration['id']
        paths=item['references'];count=min(max(1,args.views),len(paths))
        indices=[round(i*(len(paths)-1)/(count-1)) for i in range(count)] if count>1 else [0]
        for index in indices:
            with Image.open(paths[index]) as image:registry.add_reference(registration['id'],image.convert('RGB'))
    output=root/'evaluation'/'public_registration_ids.json';output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(mapping,indent=2))
    print('Seeded public-demo objects. Full reference frames may include background; inspect/crop them in the GUI if needed.')
    print('This demonstrates dataset objects, not your personal belongings. ID mapping:',output)


if __name__=='__main__':main()
