/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
#include "camera.glsl"
uniform mat4 model;

//current level
uniform sampler2D image;
uniform sampler2D normalMap;

//coarser level
uniform sampler2D imageUp;
uniform sampler2D normalMapUp;

uniform vec4 ScaleFactor, FineBlockOrig, TexSizeScale;

uniform vec2 RingSize, ViewerPos, AlphaOffset, OneOverWidth;
uniform float ZScaleFactor, ZTexScaleFactor;

//out vec3 normal;
out vec3 vertexMV;
out vec2 otc;
out float alpha;

void main() {

    // convert from grid xy to world xy coordinates
     //  ScaleFactor.xy: grid spacing of current level
     //  ScaleFactor.zw: origin of current block within world
    vec2 worldPos = in_position.xz * ScaleFactor.xy + ScaleFactor.zw;



    vec4 position = vec4(worldPos.x+ViewerPos.x,0,worldPos.y+ViewerPos.y,1);


    vec2 a = abs(worldPos)/(RingSize*0.5f);

    //TODO:: Better alpha
    alpha = clamp(max(a.x,a.y),0,1);
    alpha = clamp(alpha - 0.7f,0,1);
    alpha = alpha * (1.0/0.2f);
    alpha = clamp(alpha,0,1);
//    alpha = 0;

//    vec2 textureSize = vec2(5000,5000);
//    tc = vec2(position.xz)/textureSize;
    vec2 tc = (vec2(position.xz)+TexSizeScale.xy)*TexSizeScale.zw;
//    tc = tc * TexSizeScale.zw;
//    tc = (vec2(position.xz)+0.5f*vec2(0.01f))/vec2(0.01f);
    otc = tc;
//    tc = (in_position.xz+0.5f)*FineBlockOrig.xy+FineBlockOrig.zw;


    // sample the vertex texture
//    float height = texture(normalMap,tc).r;
    float height1 = texture(image,tc).r;
    height1 = height1*ZScaleFactor;

    float height2 = texture(imageUp,tc).r;
    height2 = height2*ZScaleFactor;



    position.y = (1-alpha)*height1+alpha*height2;

    vertexMV = vec3(view * position);
    gl_Position = viewProj * position;

//    c = texture(image,tc).rrr;

}







##GL_FRAGMENT_SHADER

#version 330
#include "camera.glsl"
uniform mat4 model;
uniform vec4 color;
uniform sampler2D normalMap;

uniform sampler2D texture1;
uniform sampler2D texture2;

//in vec3 normal;
in vec3 vertexMV;
in vec2 otc;
in float alpha;

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_position;

void main() {

}


