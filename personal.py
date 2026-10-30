"""Personal-object registration and explicit visual similarity matching."""
import hashlib
import json
from pathlib import Path
import uuid
from datetime import datetime
import numpy as np
from PIL import Image


class Registry:
    def __init__(self, directory):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'personal_objects.json'
        self.objects=json.loads(self.path.read_text()) if self.path.exists() else []
    def save(self):
        temporary=self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.objects,indent=2),encoding='utf-8')
        temporary.replace(self.path)
    def active(self):
        return [dict(item) for item in self.objects if item.get('active',True)]
    def get(self,oid):
        return next((item for item in self.objects if item['id']==oid),None)
    def create(self,name):
        name=name.strip()
        if not name:
            raise ValueError('Enter a personal object name.')
        if any(item['name'].casefold()==name.casefold() for item in self.active()):
            raise ValueError('That name is already registered. Add examples to the existing object.')
        item=dict(id=uuid.uuid4().hex,name=name,active=True,references=[],created=datetime.now().astimezone().isoformat())
        self.objects.append(item)
        self.save()
        return item
    def rename(self,oid,name):
        name=name.strip()
        if not name or any(item['id']!=oid and item['name'].casefold()==name.casefold() for item in self.active()):
            raise ValueError('Use a nonempty, unique object name.')
        self.get(oid)['name']=name
        self.save()
    def archive(self,oid):
        self.get(oid)['active']=False
        self.save()
    def remove_reference(self,oid,path):
        item=self.get(oid)
        if path in item['references']:
            item['references'].remove(path)
            self.save()
    def add_reference(self,oid,image):
        item=self.get(oid)
        if item is None or not item.get('active',True):
            raise ValueError('Choose an active registered object.')
        image=image.convert('RGB')
        if min(image.size)<32:
            raise ValueError('Crop is too small. Use a clearer, larger view of the object.')
        image.thumbnail((640,640),Image.Resampling.LANCZOS)
        digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
        folder=self.directory/'references'/oid
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/(digest+'.png')
        rel=str(path.relative_to(self.directory))
        if rel in item['references']:
            return False
        image.save(path)
        item['references'].append(rel)
        self.save()
        return True


def normalize(vectors):
    array=np.asarray(vectors,dtype=np.float32)
    if array.ndim==1:
        array=array[None,:]
    return array/np.maximum(1e-8,np.linalg.norm(array,axis=1,keepdims=True))


def rank_match(vector, gallery, settings, color=None):
    """Cosine scores are not probabilities. Similar entries require an adequate margin."""
    vector=normalize(vector)[0]
    from appearance import reference_scores
    scores=[]
    evidence={}
    for item in gallery:
        references=normalize(item['features'])
        values,deep,chroma=reference_scores(vector,references,color,item.get('colors'))
        best=int(np.argmax(values))
        score=float(values[best])
        evidence[item['id']]=dict(embedding_score=float(deep[best]),color_score=float(chroma[best]) if chroma is not None else None,reference_scores=values.tolist(),best_view=best+1)
        scores.append((score,item))
    scores.sort(key=lambda pair:pair[0],reverse=True)
    if not scores or scores[0][0]<settings['possible_threshold']:
        return dict(object_id='',name='',score=scores[0][0] if scores else 0,margin=0,strong=False,reason='No reliable reference match',candidates=[dict(id=item['id'],name=item['name'],score=score) for score,item in scores[:3]])
    score,item=scores[0]
    runner=scores[1][0] if len(scores)>1 else 0
    margin=score-runner
    threshold=item.get('threshold',settings['similarity_threshold'])
    required_margin=item.get('margin',settings['similarity_margin'])
    details=evidence[item['id']]
    # Colour helps discriminate; it may never override weak structural evidence.
    visual_ok=details['embedding_score']>=settings['similarity_threshold']
    strong=(score>=threshold and margin>=required_margin and visual_ok
            and len(item['features'])>=settings['minimum_reference_views'])
    if len(item['features'])<settings['minimum_reference_views']:
        reason='Add more distinct reference views'
    elif margin<required_margin:
        reason='Similar registered objects; review needed'
    elif score<threshold or not visual_ok:
        reason='Possible match; visual evidence below acceptance threshold'
    else:
        reason='Strong similarity; user approval required for a new track'
    return dict(object_id=item['id'],name=item['name'],score=score,margin=margin,strong=strong,reason=reason,
                threshold=threshold,required_margin=required_margin,confused=bool(item.get('confused',False) and not strong),
                **details,candidates=[dict(id=i['id'],name=i['name'],score=s) for s,i in scores[:3]])


class Embedder:
    def __init__(self,model_name,cache_dir):
        import torch
        from transformers import AutoImageProcessor,AutoModel
        self.torch=torch
        self.model_name=model_name
        # Reserve CPU capacity for camera/UI responsiveness.
        import os
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2)//2)))
        self.processor=AutoImageProcessor.from_pretrained(model_name)
        self.model=AutoModel.from_pretrained(model_name).to('cpu').eval()
        self.cache_dir=Path(cache_dir)
        self.cache_dir.mkdir(parents=True,exist_ok=True)
    def encode(self,images):
        if not images:
            return np.empty((0,0),dtype=np.float32)
        inputs=self.processor(images=images,return_tensors='pt')
        with self.torch.inference_mode():
            output=self.model(**inputs)
            features=output.last_hidden_state[:,0,:].cpu().numpy()
        return normalize(features)
    def reference(self,path):
        path=Path(path)
        digest=hashlib.sha256(path.read_bytes()+self.model_name.encode()).hexdigest()
        cache=self.cache_dir/(digest+'.npy')
        if cache.exists():
            return np.load(cache,allow_pickle=False)
        with Image.open(path) as image:
            vector=self.encode([image.convert('RGB')])[0]
        np.save(cache,vector,allow_pickle=False)
        return vector


class PersonalVision:
    def __init__(self, registry, settings):
        self.settings=settings
        self.registry=registry
        self.gallery=[]
        self.embedder=Embedder(settings['embedding_model'],registry.directory/'feature_cache')
        for item in registry.active():
            refs=[registry.directory/path for path in item['references'] if (registry.directory/path).is_file()]
            if refs:
                self.gallery.append(dict(id=item['id'],name=item['name'],paths=refs,
                                         features=np.stack([self.embedder.reference(path) for path in refs]),
                                         colors=np.stack([self.reference_color(path) for path in refs])))
        self.owl=None
        if settings['reference_search'] and self.gallery:
            from transformers import Owlv2Processor,Owlv2ForObjectDetection
            self.owl_processor=Owlv2Processor.from_pretrained(settings['reference_model'])
            self.owl=Owlv2ForObjectDetection.from_pretrained(settings['reference_model']).to('cpu').eval()
        self.scan_index=0

    @staticmethod
    def reference_color(path):
        from appearance import color_signature
        with Image.open(path) as image:
            return color_signature(image)

    def proposals(self,image):
        if self.owl is None:
            return []
        # Round-robin: one registered item/reference per frame, instead of N heavyweight passes.
        item=self.gallery[self.scan_index%len(self.gallery)]
        view=(self.scan_index//len(self.gallery))%len(item['paths'])
        self.scan_index+=1
        with Image.open(item['paths'][view]) as reference:
            inputs=self.owl_processor(images=image,query_images=reference.convert('RGB'),return_tensors='pt')
        with self.embedder.torch.inference_mode():
            output=self.owl.image_guided_detection(**inputs)
        result=self.owl_processor.post_process_image_guided_detection(output,
                threshold=self.settings['proposal_threshold'],nms_threshold=.3,
                target_sizes=self.embedder.torch.tensor([[image.height,image.width]]))[0]
        boxes=result['boxes'].cpu().tolist()
        scores=result['scores'].cpu().tolist()
        order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:4]
        return [dict(name='unclassified object',confidence=float(scores[i]),
                     box=[boxes[i][0]/image.width,boxes[i][1]/image.height,boxes[i][2]/image.width,boxes[i][3]/image.height],
                     proposal_source='Reference-guided detector') for i in order]

    def analyse(self,frame,detections):
        import cv2
        from engine import overlap
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        detections=[dict(d,proposal_source='YOLO') for d in detections]
        for proposal in self.proposals(image):
            if not any(overlap(proposal['box'],d['box'])>.45 for d in detections):
                detections.append(proposal)
        candidates=sorted(detections,key=lambda d:d['confidence'],reverse=True)[:self.settings['max_personal_candidates']]
        crops=[]
        valid=[]
        w,h=image.size
        for d in candidates:
            a,b,c,e=d['box']
            a,b,c,e=max(0,int(a*w)),max(0,int(b*h)),min(w,int(c*w)),min(h,int(e*h))
            if c-a<16 or e-b<16:
                continue
            crop=image.crop((a,b,c,e))
            gray=cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2GRAY)
            sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
            quality='Object crop is small' if min(c-a,e-b)<40 else ('Object crop appears blurred' if sharpness<25 else '')
            d['quality_warning']=quality
            crops.append(crop)
            valid.append(d)
        features=self.embedder.encode(crops)
        from appearance import color_signature
        for d,feature,crop in zip(valid,features,crops):
            match=rank_match(feature,self.gallery,self.settings,color_signature(crop))
            if d['quality_warning']:
                match['strong']=False
                match['reason']=d['quality_warning']+'; review needed'
            d['personal_match']=match
        # Embedding limits affect personal matching only, never the detector output.
        return detections
