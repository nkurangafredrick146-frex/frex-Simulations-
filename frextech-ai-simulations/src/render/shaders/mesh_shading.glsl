// Advanced Mesh Shading with PBR, Normal Mapping, and Tessellation

#version 450 core

// Vertex Shader
#ifdef VERTEX_SHADER
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aTangent;
layout(location = 4) in vec4 aColor;

out VertexData {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
    vec4 color;
    vec3 tangent;
    vec3 bitangent;
    vec4 shadowCoord;
} vOut;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat4 uNormalMatrix;
uniform mat4 uShadowMatrix;

void main() {
    vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
    vOut.worldPos = worldPos.xyz;
    vOut.normal = normalize(uNormalMatrix * aNormal);
    vOut.texCoord = aTexCoord;
    vOut.color = aColor;
    
    // Calculate TBN matrix for normal mapping
    vOut.tangent = normalize(uNormalMatrix * aTangent.xyz);
    vOut.bitangent = normalize(cross(vOut.normal, vOut.tangent)) * aTangent.w;
    
    // Shadow coordinates
    vOut.shadowCoord = uShadowMatrix * worldPos;
    
    gl_Position = uProjectionMatrix * uViewMatrix * worldPos;
}
#endif

// Tessellation Control Shader
#ifdef TESS_CONTROL_SHADER
layout(vertices = 3) out;

in VertexData {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
    vec4 color;
    vec3 tangent;
    vec3 bitangent;
    vec4 shadowCoord;
} vIn[];

out TessControlData {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
    vec4 color;
    vec3 tangent;
    vec3 bitangent;
    vec4 shadowCoord;
} tcOut[];

uniform float uTessLevelInner = 4.0;
uniform float uTessLevelOuter = 4.0;
uniform vec3 uCameraPos;
uniform float uMaxTessLevel = 64.0;
uniform float uMinTessLevel = 1.0;
uniform float uTessDistanceScale = 10.0;

float calculateTessLevel(vec3 pos0, vec3 pos1) {
    // Distance-based tessellation
    vec3 edgeCenter = (pos0 + pos1) * 0.5;
    float distance = length(uCameraPos - edgeCenter);
    
    // Screen-space adaptive tessellation
    float tessLevel = uMaxTessLevel * exp(-distance / uTessDistanceScale);
    return clamp(tessLevel, uMinTessLevel, uMaxTessLevel);
}

void main() {
    // Pass through data
    tcOut[gl_InvocationID].worldPos = vIn[gl_InvocationID].worldPos;
    tcOut[gl_InvocationID].normal = vIn[gl_InvocationID].normal;
    tcOut[gl_InvocationID].texCoord = vIn[gl_InvocationID].texCoord;
    tcOut[gl_InvocationID].color = vIn[gl_InvocationID].color;
    tcOut[gl_InvocationID].tangent = vIn[gl_InvocationID].tangent;
    tcOut[gl_InvocationID].bitangent = vIn[gl_InvocationID].bitangent;
    tcOut[gl_InvocationID].shadowCoord = vIn[gl_InvocationID].shadowCoord;
    
    // Calculate tessellation levels
    if (gl_InvocationID == 0) {
        // Edge tessellation levels
        gl_TessLevelOuter[0] = calculateTessLevel(vIn[1].worldPos, vIn[2].worldPos);
        gl_TessLevelOuter[1] = calculateTessLevel(vIn[2].worldPos, vIn[0].worldPos);
        gl_TessLevelOuter[2] = calculateTessLevel(vIn[0].worldPos, vIn[1].worldPos);
        
        // Inner tessellation level
        gl_TessLevelInner[0] = (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) / 3.0;
    }
}
#endif

// Tessellation Evaluation Shader
#ifdef TESS_EVALUATION_SHADER
layout(triangles, equal_spacing, ccw) in;

in TessControlData {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
    vec4 color;
    vec3 tangent;
    vec3 bitangent;
    vec4 shadowCoord;
} tcIn[];

out TessEvalData {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
    vec4 color;
    vec3 tangent;
    vec3 bitangent;
    vec4 shadowCoord;
} teOut;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat4 uNormalMatrix;
uniform mat4 uShadowMatrix;

// Displacement map
uniform sampler2D uDisplacementMap;
uniform float uDisplacementScale = 0.1;

vec3 interpolate(vec3 v0, vec3 v1, vec3 v2) {
    return gl_TessCoord.x * v0 + gl_TessCoord.y * v1 + gl_TessCoord.z * v2;
}

vec2 interpolate(vec2 v0, vec2 v1, vec2 v2) {
    return gl_TessCoord.x * v0 + gl_TessCoord.y * v1 + gl_TessCoord.z * v2;
}

vec4 interpolate(vec4 v0, vec4 v1, vec4 v2) {
    return gl_TessCoord.x * v0 + gl_TessCoord.y * v1 + gl_TessCoord.z * v2;
}

void main() {
    // Interpolate attributes
    vec3 worldPos = interpolate(tcIn[0].worldPos, tcIn[1].worldPos, tcIn[2].worldPos);
    vec3 normal = interpolate(tcIn[0].normal, tcIn[1].normal, tcIn[2].normal);
    vec2 texCoord = interpolate(tcIn[0].texCoord, tcIn[1].texCoord, tcIn[2].texCoord);
    vec4 color = interpolate(tcIn[0].color, tcIn[1].color, tcIn[2].color);
    vec3 tangent = interpolate(tcIn[0].tangent, tcIn[1].tangent, tcIn[2].tangent);
    vec3 bitangent = interpolate(tcIn[0].bitangent, tcIn[1].bitangent, tcIn[2].bitangent);
    vec4 shadowCoord = interpolate(tcIn[0].shadowCoord, tcIn[1].shadowCoord, tcIn[2].shadowCoord);
    
    // Apply displacement mapping
    if (uDisplacementScale > 0.0) {
        float displacement = texture(uDisplacementMap, texCoord).r * uDisplacementScale;
        worldPos += normal * displacement;
    }
    
    // Recalculate normal if displaced
    if (uDisplacementScale > 0.0) {
        // Calculate derivatives for normal reconstruction
        // This is simplified - in practice you'd compute partial derivatives
        normal = normalize(normal);
    }
    
    // Output data
    teOut.worldPos = worldPos;
    teOut.normal = normalize(normal);
    teOut.texCoord = texCoord;
    teOut.color = color;
    teOut.tangent = normalize(tangent);
    teOut.bitangent = normalize(bitangent);
    teOut.shadowCoord = shadowCoord;
    
    // Final position
    vec4 worldPos4 = vec4(worldPos, 1.0);
    gl_Position = uProjectionMatrix * uViewMatrix * worldPos4;
}
#endif

// Fragment Shader with PBR
#ifdef FRAGMENT_SHADER
in VertexData {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
    vec4 color;
    vec3 tangent;
    vec3 bitangent;
    vec4 shadowCoord;
} fIn;

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedo;
layout(location = 3) out vec4 gMetallicRoughnessAO;
layout(location = 4) out vec4 gEmissive;

// Material textures
uniform sampler2D uAlbedoMap;
uniform sampler2D uNormalMap;
uniform sampler2D uMetallicMap;
uniform sampler2D uRoughnessMap;
uniform sampler2D uAOMap;
uniform sampler2D uEmissiveMap;
uniform sampler2D uDisplacementMap;
uniform sampler2D uOpacityMap;

// Shadow map
uniform sampler2DShadow uShadowMap;
uniform float uShadowBias = 0.001;
uniform int uShadowQuality = 1; // 0: off, 1: PCF, 2: VSM, 3: ESM

// Material properties
uniform vec4 uAlbedoColor = vec4(1.0);
uniform float uMetallic = 0.0;
uniform float uRoughness = 0.5;
uniform float uAO = 1.0;
uniform vec3 uEmissiveColor = vec3(0.0);
uniform float uOpacity = 1.0;
uniform float uAlphaCutoff = 0.5;
uniform float uIOR = 1.5;
uniform vec3 uSpecularColor = vec3(0.04);

// Lights
#define MAX_LIGHTS 8
struct Light {
    vec4 position;
    vec4 direction;
    vec4 color;
    float intensity;
    float range;
    float spotAngle;
    float spotBlend;
    int type;
};

uniform Light uLights[MAX_LIGHTS];
uniform int uLightCount = 0;
uniform vec3 uAmbientLight = vec3(0.03);

// Camera
uniform vec3 uCameraPos;

// IBL (Image Based Lighting)
uniform samplerCube uIrradianceMap;
uniform samplerCube uPrefilterMap;
uniform sampler2D uBRDFLUT;
uniform float uIBLStrength = 1.0;

// Constants
const float PI = 3.14159265359;
const float Epsilon = 0.00001;

// PBR functions
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return a2 / max(denom, Epsilon);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = geometrySchlickGGX(NdotV, roughness);
    float ggx2 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Normal mapping
vec3 getNormalFromMap() {
    vec3 tangentNormal = texture(uNormalMap, fIn.texCoord).xyz * 2.0 - 1.0;

    vec3 Q1 = dFdx(fIn.worldPos);
    vec3 Q2 = dFdy(fIn.worldPos);
    vec2 st1 = dFdx(fIn.texCoord);
    vec2 st2 = dFdy(fIn.texCoord);

    vec3 N = normalize(fIn.normal);
    vec3 T = normalize(fIn.tangent);
    vec3 B = normalize(fIn.bitangent);
    
    // Ensure orthonormal basis
    T = normalize(T - dot(T, N) * N);
    B = cross(N, T);
    
    mat3 TBN = mat3(T, B, N);
    return normalize(TBN * tangentNormal);
}

// Shadow calculation
float calculateShadow(vec4 shadowCoord, vec3 normal, vec3 lightDir) {
    if (uShadowQuality == 0) return 1.0;
    
    // Perspective divide
    vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    if (projCoords.z > 1.0) return 1.0;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // Apply bias
    float bias = max(uShadowBias * (1.0 - dot(normal, lightDir)), uShadowBias);
    
    if (uShadowQuality == 1) {
        // PCF (Percentage Closer Filtering)
        float shadow = 0.0;
        vec2 texelSize = 1.0 / textureSize(uShadowMap, 0);
        
        for(int x = -1; x <= 1; ++x) {
            for(int y = -1; y <= 1; ++y) {
                float pcfDepth = texture(uShadowMap, vec3(projCoords.xy + vec2(x, y) * texelSize, currentDepth - bias));
                shadow += pcfDepth;
            }
        }
        shadow /= 9.0;
        return shadow;
    } else if (uShadowQuality == 2) {
        // VSM (Variance Shadow Maps)
        vec2 moments = texture(uShadowMap, projCoords.xy).rg;
        
        if (currentDepth <= moments.x) return 1.0;
        
        float variance = moments.y - (moments.x * moments.x);
        variance = max(variance, 0.00001);
        
        float d = currentDepth - moments.x;
        float p_max = variance / (variance + d * d);
        
        return clamp(p_max, 0.0, 1.0);
    }
    
    return 1.0;
}

void main() {
    // Alpha testing
    float opacity = texture(uOpacityMap, fIn.texCoord).r * uOpacity * fIn.color.a;
    if (opacity < uAlphaCutoff) {
        discard;
    }
    
    // Sample material textures
    vec4 albedo = texture(uAlbedoMap, fIn.texCoord) * uAlbedoColor * fIn.color;
    float metallic = texture(uMetallicMap, fIn.texCoord).r * uMetallic;
    float roughness = texture(uRoughnessMap, fIn.texCoord).r * uRoughness;
    float ao = texture(uAOMap, fIn.texCoord).r * uAO;
    vec3 emissive = texture(uEmissiveMap, fIn.texCoord).rgb * uEmissiveColor;
    
    // Normal mapping
    vec3 N = getNormalFromMap();
    vec3 V = normalize(uCameraPos - fIn.worldPos);
    vec3 R = reflect(-V, N);
    
    // Calculate reflectance at normal incidence
    vec3 F0 = mix(uSpecularColor, albedo.rgb, metallic);
    
    // Reflectance equation
    vec3 Lo = vec3(0.0);
    
    // Directional light contribution
    for (int i = 0; i < uLightCount; ++i) {
        Light light = uLights[i];
        vec3 L;
        float attenuation = 1.0;
        
        if (light.type == 0) { // Directional
            L = normalize(-light.direction.xyz);
        } else { // Point or spot
            vec3 lightVec = light.position.xyz - fIn.worldPos;
            float distance = length(lightVec);
            L = normalize(lightVec);
            
            // Attenuation
            float atten = 1.0 / (1.0 + 0.01 * distance + 0.0001 * distance * distance);
            attenuation *= atten;
            
            // Spot light
            if (light.type == 2) {
                float theta = dot(L, normalize(-light.direction.xyz));
                float epsilon = light.spotBlend;
                float intensity = clamp((theta - cos(radians(light.spotAngle))) / epsilon, 0.0, 1.0);
                attenuation *= intensity;
            }
        }
        
        // Half vector
        vec3 H = normalize(V + L);
        
        // Cook-Torrance BRDF
        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + Epsilon;
        vec3 specular = numerator / denominator;
        
        // Add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);
        vec3 radiance = light.color.rgb * light.intensity * attenuation * NdotL;
        
        // Shadow
        float shadow = 1.0;
        if (light.type == 0) { // Only directional lights cast shadows
            shadow = calculateShadow(fIn.shadowCoord, N, L);
        }
        
        Lo += (kD * albedo.rgb / PI + specular) * radiance * shadow;
    }
    
    // Ambient lighting (IBL)
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    
    vec3 irradiance = texture(uIrradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo.rgb;
    
    // Sample pre-filter map and BRDF LUT
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(uPrefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(uBRDFLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
    
    vec3 ambient = (kD * diffuse + specular) * ao * uIBLStrength;
    
    // Emissive
    vec3 emissiveContribution = emissive;
    
    // Final color
    vec3 color = ambient + Lo + emissiveContribution;
    
    // Tone mapping
    color = color / (color + vec3(1.0));
    
    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));
    
    // Output to G-buffer
    gPosition = vec4(fIn.worldPos, 1.0);
    gNormal = vec4(N, 1.0);
    gAlbedo = vec4(albedo.rgb, opacity);
    gMetallicRoughnessAO = vec4(metallic, roughness, ao, 1.0);
    gEmissive = vec4(emissive, 1.0);
    
    // Debug outputs
    #ifdef DEBUG_NORMALS
        gAlbedo = vec4(N * 0.5 + 0.5, 1.0);
    #endif
    
    #ifdef DEBUG_ALBEDO
        gAlbedo = vec4(albedo.rgb, 1.0);
    #endif
    
    #ifdef DEBUG_METALLIC
        gAlbedo = vec4(vec3(metallic), 1.0);
    #endif
    
    #ifdef DEBUG_ROUGHNESS
        gAlbedo = vec4(vec3(roughness), 1.0);
    #endif
    
    #ifdef DEBUG_AO
        gAlbedo = vec4(vec3(ao), 1.0);
    #endif
    
    #ifdef DEBUG_EMISSIVE
        gAlbedo = vec4(emissive, 1.0);
    #endif
}
#endif

// Deferred Shading Fragment Shader
#ifdef DEFERRED_FRAGMENT_SHADER
in vec2 vTexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gMetallicRoughnessAO;
uniform sampler2D gEmissive;

uniform vec3 uCameraPos;
uniform Light uLights[MAX_LIGHTS];
uniform int uLightCount;
uniform vec3 uAmbientLight;

out vec4 fragColor;

void main() {
    // Read from G-buffer
    vec3 worldPos = texture(gPosition, vTexCoord).rgb;
    vec3 N = texture(gNormal, vTexCoord).rgb;
    vec3 albedo = texture(gAlbedo, vTexCoord).rgb;
    float opacity = texture(gAlbedo, vTexCoord).a;
    float metallic = texture(gMetallicRoughnessAO, vTexCoord).r;
    float roughness = texture(gMetallicRoughnessAO, vTexCoord).g;
    float ao = texture(gMetallicRoughnessAO, vTexCoord).b;
    vec3 emissive = texture(gEmissive, vTexCoord).rgb;
    
    vec3 V = normalize(uCameraPos - worldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    
    // Lighting calculation (same as forward rendering)
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < uLightCount; ++i) {
        Light light = uLights[i];
        vec3 L;
        float attenuation = 1.0;
        
        if (light.type == 0) {
            L = normalize(-light.direction.xyz);
        } else {
            vec3 lightVec = light.position.xyz - worldPos;
            float distance = length(lightVec);
            L = normalize(lightVec);
            float atten = 1.0 / (1.0 + 0.01 * distance + 0.0001 * distance * distance);
            attenuation *= atten;
            
            if (light.type == 2) {
                float theta = dot(L, normalize(-light.direction.xyz));
                float epsilon = light.spotBlend;
                float intensity = clamp((theta - cos(radians(light.spotAngle))) / epsilon, 0.0, 1.0);
                attenuation *= intensity;
            }
        }
        
        vec3 H = normalize(V + L);
        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + Epsilon;
        vec3 specular = numerator / denominator;
        
        float NdotL = max(dot(N, L), 0.0);
        vec3 radiance = light.color.rgb * light.intensity * attenuation * NdotL;
        
        Lo += (kD * albedo / PI + specular) * radiance;
    }
    
    vec3 ambient = uAmbientLight * albedo * ao;
    vec3 color = ambient + Lo + emissive;
    
    // Tone mapping and gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    
    fragColor = vec4(color, opacity);
}
#endif
