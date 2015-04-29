#pragma once

#include "libhello/opengl/mesh_object.h"

#include "libhello/geometry/sphere.h"
#include "libhello/geometry/plane.h"
#include "libhello/geometry/triangle_mesh_generator.h"

#include "libhello/rendering/lighting/light.h"
#include "libhello/camera/camera.h"
#include "libhello/opengl/framebuffer.h"

class DirectionalLightShader : public LightShader{
public:
    GLuint location_direction, location_ambientIntensity;


    DirectionalLightShader(const string &multi_file) : LightShader(multi_file){}
    virtual void checkUniforms();
    void uploadDirection(vec3 &direction);
    void uploadAmbientIntensity(float i);

};

class DirectionalLight :  public Light
{
protected:


    vec3 direction = vec3(0,-1,0);
    float range = 20.0f;
    float ambientIntensity = 0.2f;

public:
     OrthographicCamera cam;
    const mat4 *view;
//    static void createMesh();
    DirectionalLight();
    virtual ~DirectionalLight(){}

    void bindUniforms(DirectionalLightShader& shader, Camera* cam);

    virtual void createShadowMap(int resX, int resY) override;
    void setDirection(const vec3 &dir);
    void setFocus(const vec3 &pos);
    void setAmbientIntensity(float ai);


};


