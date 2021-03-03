#include "image.h"
#include "kdtree.h"
#include "ray.h"
#include "raytracer.h"
#include "scene_types.h"
#include <stdio.h>

#include <glm/gtc/epsilon.hpp>

#define EOR 1.0f
#define DEEPMAX 10.0f
#define PI 3.14159265358979323846f

const float acne_eps = 1e-4;

bool intersectPlane(Ray *ray, Intersection *intersection, Object *obj)
{
    float t;
    vec3 d = ray->dir;  // distance a l'origine
    vec3 n = obj->geom.plane.normal; // normale

    if(dot(d,n)==0.0f)
        return false; // pas de solution car le rayon est parallele au plan

    else {
        t = -(dot(ray->orig,n) +obj->geom.plane.dist)/dot(n,d);
        if(!(t >= ray->tmin && t <=ray->tmax)) // vérification si t est entre tmin et tmax pour le rayon
          return false;
    }
    // mise à jour des paramètres de la structure Intersection et de Ray
    ray->tmax=t;
    intersection->normal = n;
    intersection->position=rayAt(*ray,t);
    intersection->mat=&(obj->mat);

    return true;

}

bool intersectSphere(Ray *ray, Intersection *intersection, Object *obj)
{

    point3 O = ray->orig; // origine du rayon
    vec3 d = ray->dir;  // direction du rayon
    point3 C = obj->geom.sphere.center; // centre de la sphère
    float R = obj->geom.sphere.radius;  //  rayon de la sphère

    // partie résolution de l'équation du second degre
    float a = 1;
    float b = 2*dot(d, (O - C));
    float c = dot((O - C), (O - C)) - R*R;

    float t;
    bool inter;
    float delta = (b*b) - (4*a*c);

    if(delta>0){ // deux solutions
        float x1 = (-b -sqrt(delta))/(a*2); // calcul de la premiere solution
        float x2 = (-b+sqrt(delta))/(a*2);  // calcul de la deuxieme solution
        if((x1>=0) && (x2>=0)){  // on garde la plus petite solution supérieur ou égale a 0
          if(x1>x2) t = x2;
          else t = x1;
        }
        else if ((x1<0) && (x2>=0)) t = x2;
        else if ((x1>=0) && (x2<0)) t = x1;
        else
          inter=false;

        inter = ((t<=ray->tmax) && (t>=ray->tmin));
    }
    else if(delta==0){ // une solution car le rayon est tangent à la sphère
        t = (-b)/(2*a);
        inter = ((t<=ray->tmax) && (t>=ray->tmin));
    }
    else inter = false; // pas de solution car il n'y a pas d'intersection

    if (inter){
        // mise à jour des paramètres de la structure Intersection et de Ray
        intersection->mat = &obj->mat ;
        intersection->position = rayAt (*ray, t);
        intersection->normal = normalize(intersection->position-C);
        ray->tmax = t;
    }

    return inter;
}

bool intersectScene(const Scene *scene, Ray *ray, Intersection *intersection) {
  bool hasIntersection = false;

  for(Object *obj : scene->objects){
    if(obj->geom.type==SPHERE)
      if(intersectSphere(ray, intersection, obj))
        hasIntersection=true;
    if(obj->geom.type==PLANE)
      if(intersectPlane(ray, intersection, obj))
        hasIntersection=true;

  }
  return hasIntersection;  // vérifie s’il y a une intersection entre le rayon et chaque objet de la scène
}

/* ---------------------------------------------------------------------------
 */
/*
 *  The following functions are coded from Cook-Torrance bsdf model
 *description and are suitable only
 *  for rough dielectrics material (RDM. Code has been validated with Mitsuba
 *renderer)
 */

// Shadowing and masking function. Linked with the NDF. Here, Smith function,
// suitable for Beckmann NDF
float RDM_chiplus(float c) { return (c > 0.f) ? 1.f : 0.f; }

/** Normal Distribution Function : Beckmann
 * NdotH : Norm . Half
 */
float RDM_Beckmann(float NdotH, float alpha) {

    if (NdotH>0.0f){
    	float cosCarre = NdotH*NdotH;
    	float tang = (1.0f-cosCarre)/(cosCarre);
    	return expf(-(tang)/(alpha*alpha))/((float)M_PI*alpha*alpha*cosCarre*cosCarre);
    }
    else
    	return 0.0f;
}

// Fresnel term computation. Implantation of the exact computation. we can use
// the Schlick approximation
// LdotH : Light . Half
float RDM_Fresnel(float LdotH, float extIOR, float intIOR) {

  float f, Rs, Rp, sincarre = (extIOR/intIOR)*(extIOR/intIOR)*(1-(LdotH*LdotH));
  if (sincarre>1.0f)
		return 1.0f;

  Rs = powf((extIOR*LdotH-intIOR*sqrtf((1-sincarre)))/((extIOR*LdotH+intIOR*sqrtf((1-sincarre)))),2);
  Rp = powf((extIOR*sqrtf((1-sincarre))-intIOR*LdotH)/((extIOR*sqrtf((1-sincarre))+intIOR*LdotH)),2);
  f = 0.5f*(Rs+Rp);

  return f;
}


// DdotH : Dir . Half
// HdotN : Half . Norm
float RDM_G1(float DdotH, float DdotN, float alpha) {

  float k,cos,tan;
  k = DdotH/DdotN;
  cos = DdotN;
  tan = sqrtf(1.0f-cos*cos)/cos;
  float b = 1.0f/(alpha*tan);
  if(k>0){
 	  if(b<1.6f){
 	    float g1 = RDM_chiplus(k)*((3.535*b+2.181*b*b)/(1.0f+2.276*b+2.577*b*b));
	    return g1;
 	  }
    else
      return RDM_chiplus(k);
  }
  else
    return 0.0f;
}

// LdotH : Light . Half
// LdotN : Light . Norm
// VdotH : View . Half
// VdotN : View . Norm
float RDM_Smith(float LdotH, float LdotN, float VdotH, float VdotN, float alpha) {
  return dot(RDM_G1(LdotH,LdotN,alpha),RDM_G1(VdotH,VdotN,alpha));
}

// Specular term of the Cook-torrance bsdf
// LdotH : Light . Half
// NdotH : Norm . Half
// VdotH : View . Half
// LdotN : Light . Norm
// VdotN : View . Norm
color3 RDM_bsdf_s(float LdotH, float NdotH, float VdotH, float LdotN, float VdotN, Material *m) {
  return RDM_Smith(LdotH,LdotN,VdotH,VdotN,m->roughness)*RDM_Fresnel(LdotH,EOR,m->IOR)*RDM_Beckmann(NdotH,m->roughness)*m->specularColor/(4*LdotN*VdotN);
}

// diffuse term of the cook torrance bsdf
color3 RDM_bsdf_d(Material *m) {
  return ((m->diffuseColor)/(float)M_PI);
}

// The full evaluation of bsdf(wi, wo) * cos (thetai)
// LdotH : Light . Half
// NdotH : Norm . Half
// VdotH : View . Half
// LdotN : Light . Norm
// VdtoN : View . Norm
// compute bsdf * cos(Oi)
color3 RDM_bsdf(float LdotH, float NdotH, float VdotH, float LdotN, float VdotN,
                Material *m) {
  return RDM_bsdf_s(LdotH,NdotH,VdotH,LdotN,VdotN,m)+RDM_bsdf_d(m);
}

/*
color3 shade(vec3 n, vec3 v, vec3 l, color3 lc, Material *mat) {
  color3 ret = (mat->diffuseColor/PI)*dot(l,n)*lc;
  return ret;
}
*/
// shade BSDF
color3 shade(vec3 n, vec3 v, vec3 l, color3 lc, Material *mat) {

  vec3 h = normalize(l+v);

  float LdotH = dot(l,h);
  float NdotH = dot(n,h);
  float VdotH = dot(v,h);
  float LdotN = dot(l,n);
  float VdotN = dot(v,n);

  color3 ret = lc*RDM_bsdf(LdotH,NdotH,VdotH,LdotN,VdotN,mat)*LdotN;
  return ret;
}

//! if tree is not null, use intersectKdTree to compute the intersection instead
//! of intersect scene
/*
// trace_ray SHADING
color3 trace_ray(Scene * scene, Ray *ray, KdTree *tree) {
  color3 ret = color3(0,0,0);
  Intersection intersection;
  //Verifier que le rayon rencongtre au moins un objet de la scène
  if(!intersectScene(scene,ray,&intersection))
    ret = scene->skyColor;
  else {
    // ret = (0.5f*(intersection.normal)) + 0.5f; //test du lancer de rayon
    size_t lightCount = scene->lights.size();
    for(size_t i=0; i<lightCount; i++){
      vec3 v = -ray->dir;
      vec3 l = normalize(scene->lights[i]->position - intersection.position);
      ret += shade(intersection.normal,v,l,scene->lights[i]->color,intersection.mat);
    }
  }
  return ret;
}
*/
/*
// trace_ray OMBRES
color3 trace_ray(Scene *scene, Ray *ray, KdTree *tree) {

  color3 ret = color3(0, 0, 0);
  Intersection intersection;

  if(!(intersectScene(scene, ray, &intersection) )){
    ret= scene->skyColor;
  }
  else{
    for(Light *light: scene->lights){
      vec3 l = normalize(light->position - intersection.position);
      Ray ombre;
      Intersection iOmbre;
      rayInit(&ombre, intersection.position + acne_eps * l, l,0, length(light->position - intersection.position));
      if(!intersectScene(scene, &ombre, &iOmbre)){
        vec3 v = -(ray->dir);
        ret += shade(intersection.normal, v, l, light->color, intersection.mat);
      }
      else ret+=color3(0,0,0);
    }
  }
  return ret;
}

*/
// trace_ray REFLEXION
color3 trace_ray(Scene * scene, Ray *ray, KdTree *tree) {
  color3 ret = color3(0,0,0);
  color3 cd = color3(0,0,0);
  color3 cr = color3(0,0,0);
  Intersection intersection;
  Intersection intersectOmbre;
  Ray reflectRay;
  vec3 reflectDir;
  float fresnel = 0.f;

  // on verifie que le rayon rencontre au moins un objet de la scène
  if(!intersectScene(scene,ray,&intersection))
    ret = scene->skyColor;

  else {
    size_t lightCount = scene->lights.size();
    for(size_t i=0; i<lightCount; i++){
      Ray ombreRay;
      point3 ip = intersection.position;
      vec3 l = normalize(scene->lights[i]->position - ip);
      point3 o = ip + (acne_eps*l);

      // création d'un rayon d'ombre
      rayInit(&ombreRay, intersection.position+acne_eps*l, l, 0, distance(scene->lights[i]->position, ip));

      if((!intersectScene(scene,&ombreRay,&intersectOmbre))){
        vec3 v = -ray->dir;
        cd += shade(intersection.normal,v,l,scene->lights[i]->color,intersection.mat);
      }

    }
    // calcul de cr
    if(ray->depth < 10){
      // création du rayon de reflexion
      reflectDir = reflect(ray->dir,intersection.normal);
      float LdotH = dot(reflectDir,normalize(-ray->dir+reflectDir));
      rayInit(&reflectRay,intersection.position,reflectDir,acne_eps,10000,ray->depth+1);
      fresnel = RDM_Fresnel(LdotH,1.f,intersection.mat->IOR);
      reflectRay.depth = ray->depth+1;
      cr = trace_ray(scene,&reflectRay,tree);
    }
    ret = cd + fresnel*cr*intersection.mat->specularColor;
  }

  return ret;
}



void renderImage(Image *img, Scene *scene) {
  //! This function is already operational, you might modify it for antialiasing
  //! and kdtree initializaion
  float aspect = 1.f / scene->cam.aspect;

  KdTree *tree = NULL;

//! \todo initialize KdTree

  float delta_y = 1.f / (img->height * 0.5f);   //! one pixel size
  vec3 dy = delta_y * aspect * scene->cam.ydir; //! one pixel step
  vec3 ray_delta_y = (0.5f - img->height * 0.5f) / (img->height * 0.5f) * aspect * scene->cam.ydir;

  float delta_x = 1.f / (img->width * 0.5f);
  vec3 dx = delta_x * scene->cam.xdir;
  vec3 ray_delta_x = (0.5f - img->width * 0.5f) / (img->width * 0.5f) * scene->cam.xdir;


  for (size_t j = 0; j < img->height; j++) {
    if (j != 0)
      printf("\033[A\r");
    float progress = (float)j / img->height * 100.f;
    printf("progress\t[");
    int cpt = 0;
    for (cpt = 0; cpt < progress; cpt += 5)
      printf(".");
    for (; cpt < 100; cpt += 5)
      printf(" ");
    printf("]\n");
    #pragma omp parallel for
    for (size_t i = 0; i < img->width; i++) {
      color3 *ptr = getPixelPtr(img, i, j);
      vec3 ray_dir = scene->cam.center + ray_delta_x + ray_delta_y + float(i) * dx + float(j) * dy;
      Ray rx;
      rayInit(&rx, scene->cam.position, normalize(ray_dir));
      *ptr = trace_ray(scene, &rx, tree);

    }
  }
}
