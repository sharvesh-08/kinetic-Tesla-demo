const THREE = window.THREE;
const OrbitControls = (window.THREE && window.THREE.OrbitControls) || window.OrbitControls;

// ---------------------------------------------------------------------------
// 1. Scene, Camera, Renderer & Studio Tone Mapping Setup
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b14);

const camera = new THREE.PerspectiveCamera(
    40,
    (window.innerWidth || 800) / (window.innerHeight || 600),
    0.1,
    1000
);
camera.position.set(3.6, 2.2, 3.6);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth || 800, window.innerHeight || 600);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
document.body.appendChild(renderer.domElement);

// Expose globally for index.html auto-resize
window._engineRenderer = renderer;
window._engineCamera = camera;

// Orbit Controls for 360° Inspection
let controls;
try {
    if (typeof OrbitControls === "function") {
        controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0.65, 0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2 + 0.12;
        controls.minDistance = 1.0;
        controls.maxDistance = 10.0;
    }
} catch (e) {
    console.warn("OrbitControls fallback:", e);
}



// ---------------------------------------------------------------------------
// 2. Procedural Studio HDR Environment Map Generator (High Specular Reflections)
// ---------------------------------------------------------------------------
function generateStudioEnvironmentTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");

    // Studio Background Gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, 256);
    bgGrad.addColorStop(0, "#0b1120");
    bgGrad.addColorStop(0.5, "#1e293b");
    bgGrad.addColorStop(1, "#070a12");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, 512, 256);

    // Main Softbox Key Light Reflection Panel
    const keyPanel = ctx.createRadialGradient(256, 35, 5, 256, 35, 120);
    keyPanel.addColorStop(0, "rgba(255, 255, 255, 1.0)");
    keyPanel.addColorStop(0.35, "rgba(240, 248, 255, 0.85)");
    keyPanel.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = keyPanel;
    ctx.beginPath(); ctx.arc(256, 35, 120, 0, Math.PI * 2); ctx.fill();

    // Side Fill Light Softbox Reflection
    const fillPanel = ctx.createRadialGradient(90, 140, 5, 90, 140, 90);
    fillPanel.addColorStop(0, "rgba(186, 230, 253, 0.75)");
    fillPanel.addColorStop(1, "rgba(186, 230, 253, 0)");
    ctx.fillStyle = fillPanel;
    ctx.beginPath(); ctx.arc(90, 140, 90, 0, Math.PI * 2); ctx.fill();

    // Warm Rim Light Highlight Reflection
    const rimPanel = ctx.createRadialGradient(420, 120, 5, 420, 120, 85);
    rimPanel.addColorStop(0, "rgba(254, 215, 170, 0.85)");
    rimPanel.addColorStop(1, "rgba(254, 215, 170, 0)");
    ctx.fillStyle = rimPanel;
    ctx.beginPath(); ctx.arc(420, 120, 85, 0, Math.PI * 2); ctx.fill();

    const envTex = new THREE.CanvasTexture(canvas);
    envTex.mapping = THREE.EquirectangularReflectionMapping;
    return envTex;
}

const studioEnvMap = generateStudioEnvironmentTexture();
scene.environment = studioEnvMap;

// Studio 3-Point Directional Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
keyLight.position.set(4.5, 8.0, 5.5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.width = 2048;
keyLight.shadow.mapSize.height = 2048;
keyLight.shadow.bias = -0.0001;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
fillLight.position.set(-5, 4.5, -3.5);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xf97316, 1.1);
rimLight.position.set(1.5, 6.5, -6.0);
scene.add(rimLight);

// Floor Grid & Shadow Plane
const gridHelper = new THREE.GridHelper(10, 20, 0x38bdf8, 0x1e293b);
gridHelper.position.y = 0.0;
scene.add(gridHelper);

// ---------------------------------------------------------------------------
// Environmental Background Particle System & Dynamic Lighting Setup
// ---------------------------------------------------------------------------
const particleCount = 450;
const particleGeo = new THREE.BufferGeometry();
const particlePositions = new Float32Array(particleCount * 3);
const particleVelocities = new Float32Array(particleCount * 3);

for (let i = 0; i < particleCount; i++) {
    particlePositions[i * 3] = (Math.random() - 0.5) * 20;
    particlePositions[i * 3 + 1] = Math.random() * 12 - 2;
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 20;

    particleVelocities[i * 3] = (Math.random() - 0.5) * 0.03;
    particleVelocities[i * 3 + 1] = -Math.random() * 0.08 - 0.02;
    particleVelocities[i * 3 + 2] = (Math.random() - 0.5) * 0.03;
}
particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));

const particleMat = new THREE.PointsMaterial({
    color: 0x38bdf8,
    size: 0.06,
    transparent: true,
    opacity: 0.35,
    blending: THREE.AdditiveBlending
});

const envParticleSystem = new THREE.Points(particleGeo, particleMat);
scene.add(envParticleSystem);

// ---------------------------------------------------------------------------
// Realistic Photorealistic Environment Background Canvas Generators
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Redesigned Photorealistic WebGL Environment Sky Generators & Visuals
// ---------------------------------------------------------------------------

// 1. High-Altitude Stratospheric Sky (35,000+ ft)
function createHighAltitudeSkyTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024; canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Stratospheric Sky (Deep space top -> vibrant atmospheric blue limb)
    const sky = ctx.createLinearGradient(0, 0, 0, 360);
    sky.addColorStop(0.0, "#01010a");
    sky.addColorStop(0.20, "#080d2a");
    sky.addColorStop(0.50, "#17357a");
    sky.addColorStop(0.80, "#2563eb");
    sky.addColorStop(1.0, "#38bdf8");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, 1024, 360);

    // Distant Snow Summit Peak
    ctx.fillStyle = "#93c5fd";
    ctx.beginPath();
    ctx.moveTo(340, 360);
    ctx.lineTo(512, 140);
    ctx.lineTo(684, 360);
    ctx.closePath(); ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(475, 180);
    ctx.lineTo(512, 140);
    ctx.lineTo(540, 180);
    ctx.closePath(); ctx.fill();

    // 3D Volumetric Cloud Ocean Deck beneath altitude level
    for (let i = 0; i < 16; i++) {
        const cx = i * 70 + 15;
        const cy = 340 + Math.sin(i * 1.3) * 16;
        const r = 85 + (i % 4) * 22;
        const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, r);
        grad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
        grad.addColorStop(0.5, "rgba(219, 234, 254, 0.80)");
        grad.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    }

    ctx.fillStyle = "#040714";
    ctx.fillRect(0, 385, 1024, 127);

    const tex = new THREE.CanvasTexture(canvas);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    return tex;
}

// 2. Desert Heatwave & Scorching Sunburst (45°C–50°C)
function createDesertSkyTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024; canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Scorching Sunset/Midday Sky Gradient
    const sky = ctx.createLinearGradient(0, 0, 0, 330);
    sky.addColorStop(0.0, "#360e02");
    sky.addColorStop(0.20, "#7c2d12");
    sky.addColorStop(0.45, "#c2410c");
    sky.addColorStop(0.70, "#ea580c");
    sky.addColorStop(0.88, "#f97316");
    sky.addColorStop(1.0, "#fef08a");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, 1024, 330);

    // Glowing Desert Sun Arc
    const sunGrad = ctx.createRadialGradient(512, 280, 10, 512, 280, 180);
    sunGrad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
    sunGrad.addColorStop(0.3, "rgba(254, 240, 138, 0.7)");
    sunGrad.addColorStop(1, "rgba(234, 88, 12, 0)");
    ctx.fillStyle = sunGrad;
    ctx.beginPath(); ctx.arc(512, 280, 180, 0, Math.PI * 2); ctx.fill();

    // Warm Thermal Sunset Clouds
    for (let i = 0; i < 9; i++) {
        const cx = 90 + i * 115;
        const cy = 60 + (i % 3) * 35;
        const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 150);
        grad.addColorStop(0, "rgba(254, 215, 170, 0.50)");
        grad.addColorStop(0.5, "rgba(194, 65, 12, 0.25)");
        grad.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, 150, 0, Math.PI * 2); ctx.fill();
    }

    // Rolling Desert Dunes Layers
    ctx.fillStyle = "#c2410c";
    ctx.beginPath(); ctx.moveTo(0, 330);
    for (let x = 0; x <= 1024; x += 20) {
        ctx.lineTo(x, 305 + Math.sin(x * 0.008) * 35 + Math.cos(x * 0.015) * 18);
    }
    ctx.lineTo(1024, 512); ctx.lineTo(0, 512); ctx.closePath(); ctx.fill();

    ctx.fillStyle = "#9a3412";
    ctx.beginPath(); ctx.moveTo(0, 350);
    for (let x = 0; x <= 1024; x += 15) {
        ctx.lineTo(x, 335 + Math.sin(x * 0.012 + 1.2) * 28 + Math.cos(x * 0.022) * 12);
    }
    ctx.lineTo(1024, 512); ctx.lineTo(0, 512); ctx.closePath(); ctx.fill();

    ctx.fillStyle = "#451a03";
    ctx.beginPath(); ctx.moveTo(0, 395);
    for (let x = 0; x <= 1024; x += 15) {
        ctx.lineTo(x, 380 + Math.sin(x * 0.018 + 2.8) * 20);
    }
    ctx.lineTo(1024, 512); ctx.lineTo(0, 512); ctx.closePath(); ctx.fill();

    const tex = new THREE.CanvasTexture(canvas);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    return tex;
}

// 3. Sub-Zero Arctic Freeze & Aurora Borealis (-30°C)
function createArcticSkyTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024; canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Arctic Polar Night Sky
    const sky = ctx.createLinearGradient(0, 0, 0, 330);
    sky.addColorStop(0.0, "#010814");
    sky.addColorStop(0.30, "#06182e");
    sky.addColorStop(0.65, "#0b3452");
    sky.addColorStop(1.0, "#38bdf8");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, 1024, 330);

    // Glowing Aurora Borealis (Northern Lights) Ribbons
    for (let a = 0; a < 4; a++) {
        const aurGrad = ctx.createLinearGradient(0, 40 + a * 30, 1024, 120 + a * 30);
        aurGrad.addColorStop(0.0, "rgba(52, 211, 153, 0)");
        aurGrad.addColorStop(0.25, "rgba(52, 211, 153, 0.65)");
        aurGrad.addColorStop(0.60, "rgba(56, 189, 248, 0.55)");
        aurGrad.addColorStop(0.85, "rgba(168, 85, 247, 0.40)");
        aurGrad.addColorStop(1.0, "rgba(52, 211, 153, 0)");

        ctx.fillStyle = aurGrad;
        ctx.beginPath();
        ctx.moveTo(0, 80 + a * 25);
        for (let x = 0; x <= 1024; x += 30) {
            const y = 80 + a * 25 + Math.sin(x * 0.008 + a) * 45 + Math.cos(x * 0.015) * 20;
            ctx.lineTo(x, y);
        }
        ctx.lineTo(1024, 180 + a * 25);
        for (let x = 1024; x >= 0; x -= 30) {
            const y = 130 + a * 25 + Math.sin(x * 0.008 + a) * 45;
            ctx.lineTo(x, y);
        }
        ctx.closePath(); ctx.fill();
    }

    // Snow-Capped Glacial Mountains
    ctx.fillStyle = "#e0f2fe";
    ctx.beginPath(); ctx.moveTo(0, 330);
    const peaks = [0, 240, 120, 180, 230, 260, 370, 170, 490, 235, 630, 160, 790, 245, 930, 180, 1024, 240];
    ctx.lineTo(peaks[0], peaks[1]);
    for (let i = 2; i < peaks.length; i += 2) {
        ctx.lineTo(peaks[i], peaks[i + 1]);
    }
    ctx.lineTo(1024, 512); ctx.lineTo(0, 512); ctx.closePath(); ctx.fill();

    // Glacial Ice Lake (Foreground)
    ctx.fillStyle = "#0284c7";
    ctx.beginPath(); ctx.moveTo(0, 320);
    for (let x = 0; x <= 1024; x += 25) {
        ctx.lineTo(x, 325 + Math.sin(x * 0.01) * 15);
    }
    ctx.lineTo(1024, 512); ctx.lineTo(0, 512); ctx.closePath(); ctx.fill();

    ctx.fillStyle = "#031826";
    ctx.fillRect(0, 375, 1024, 137);

    const tex = new THREE.CanvasTexture(canvas);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    return tex;
}

// 4. Convective Severe Storm & Lightning Tempest
function createStormSkyTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024; canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Thunderstorm Tempest Sky
    const sky = ctx.createLinearGradient(0, 0, 0, 350);
    sky.addColorStop(0.0, "#020617");
    sky.addColorStop(0.30, "#0b1329");
    sky.addColorStop(0.60, "#1e293b");
    sky.addColorStop(1.0, "#334155");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, 1024, 350);

    // Heavy Supercell Storm Cloud Layers
    for (let i = 0; i < 11; i++) {
        const cx = i * 100 + 20;
        const cy = 80 + (i % 3) * 45;
        const r = 150 + (i % 2) * 50;
        const grad = ctx.createRadialGradient(cx, cy, 20, cx, cy, r);
        grad.addColorStop(0, "rgba(30, 41, 59, 0.90)");
        grad.addColorStop(0.6, "rgba(15, 23, 42, 0.95)");
        grad.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    }

    // Glowing Primary Lightning Bolt Strike
    ctx.strokeStyle = "#f0fdf4";
    ctx.lineWidth = 3.5;
    ctx.shadowColor = "#38bdf8";
    ctx.shadowBlur = 22;
    ctx.beginPath();
    ctx.moveTo(680, 50);
    ctx.lineTo(650, 130);
    ctx.lineTo(695, 175);
    ctx.lineTo(660, 255);
    ctx.lineTo(685, 335);
    ctx.stroke();

    // Secondary Lightning Branch
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(695, 175);
    ctx.lineTo(735, 225);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Dark Rain Curtain Horizon
    ctx.fillStyle = "#050910";
    ctx.fillRect(0, 355, 1024, 157);

    const tex = new THREE.CanvasTexture(canvas);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    return tex;
}

// 5. Aerospace Digital Twin Test Cell Lab (ISA Standard)
function createISASkyTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024; canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Clean Aerospace Studio Backdrop
    const sky = ctx.createLinearGradient(0, 0, 0, 512);
    sky.addColorStop(0.0, "#030712");
    sky.addColorStop(0.5, "#0b1329");
    sky.addColorStop(1.0, "#030712");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, 1024, 512);

    // Glowing Neon Blue Softbox Arc
    const softbox = ctx.createRadialGradient(512, 100, 10, 512, 100, 320);
    softbox.addColorStop(0, "rgba(56, 189, 248, 0.35)");
    softbox.addColorStop(0.6, "rgba(14, 165, 233, 0.10)");
    softbox.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = softbox;
    ctx.beginPath(); ctx.arc(512, 100, 320, 0, Math.PI * 2); ctx.fill();

    const tex = new THREE.CanvasTexture(canvas);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    return tex;
}

const envTextures = {
    HIGH_HEAT_DESERT: createDesertSkyTexture(),
    ARCTIC_FREEZE: createArcticSkyTexture(),
    HIGH_ALTITUDE_THIN_AIR: createHighAltitudeSkyTexture(),
    LIVE_STORM_TURBULENCE: createStormSkyTexture(),
    ISA_STANDARD: createISASkyTexture(),
};

let currentEnvProfile = "";
let stormFlashTimer = 0.0;

function updateEnvironmentVisuals(profile) {
    if (!profile) return;
    const rawKey = String(profile).toUpperCase().trim();
    
    let resolvedProfile = "ISA_STANDARD";
    if (rawKey.includes("ALT") || rawKey.includes("THIN")) {
        resolvedProfile = "HIGH_ALTITUDE_THIN_AIR";
    } else if (rawKey.includes("ENDUR") || rawKey.includes("ARCTIC") || rawKey.includes("FREEZE")) {
        resolvedProfile = "ARCTIC_FREEZE";
    } else if (rawKey.includes("HOT") || rawKey.includes("HEAT") || rawKey.includes("DESERT")) {
        resolvedProfile = "HIGH_HEAT_DESERT";
    } else if (rawKey.includes("RAPID") || rawKey.includes("TRANS") || rawKey.includes("STORM") || rawKey.includes("TURB")) {
        resolvedProfile = "LIVE_STORM_TURBULENCE";
    } else if (envTextures[rawKey]) {
        resolvedProfile = rawKey;
    }

    currentEnvProfile = resolvedProfile;

    // Completely disable static CSS photo background on document body
    document.body.style.backgroundImage = 'none';
    document.body.style.backgroundColor = '#080c14';

    // Apply 3D WebGL Equirectangular Environment Background Texture
    const bgTexture = envTextures[resolvedProfile] || envTextures.ISA_STANDARD;
    scene.background = bgTexture;

    // Adapt Floor Grid Helper & Environmental Lighting & Particles per profile
    if (resolvedProfile === "HIGH_HEAT_DESERT") {
        scene.fog = new THREE.FogExp2(0x3b1103, 0.028);
        particleMat.color.setHex(0xf97316);
        particleMat.opacity = 0.60;
        particleMat.size = 0.08;
        ambientLight.color.setHex(0xffedd5);
        ambientLight.intensity = 1.15;
        keyLight.color.setHex(0xffedd5);
        rimLight.color.setHex(0xf97316);

        // Warm Bronze Desert Sand Grid
        if (typeof gridHelper !== "undefined" && gridHelper.material) {
            gridHelper.material.color.setHex(0xf97316);
        }
    } else if (resolvedProfile === "ARCTIC_FREEZE") {
        scene.fog = new THREE.FogExp2(0x0b3452, 0.032);
        particleMat.color.setHex(0xe0f2fe);
        particleMat.opacity = 0.70;
        particleMat.size = 0.07;
        ambientLight.color.setHex(0xe0f2fe);
        ambientLight.intensity = 1.05;
        keyLight.color.setHex(0xf0f9ff);
        rimLight.color.setHex(0x38bdf8);

        // Frosted Blue Ice Sheet Grid
        if (typeof gridHelper !== "undefined" && gridHelper.material) {
            gridHelper.material.color.setHex(0x38bdf8);
        }
    } else if (resolvedProfile === "LIVE_STORM_TURBULENCE") {
        scene.fog = new THREE.FogExp2(0x0b1624, 0.038);
        particleMat.color.setHex(0x38bdf8);
        particleMat.opacity = 0.75;
        particleMat.size = 0.09;
        ambientLight.color.setHex(0x1e293b);
        ambientLight.intensity = 0.75;
        keyLight.color.setHex(0xf8fafc);
        rimLight.color.setHex(0xa855f7);

        // Wet Dark Reflective Tarmac Grid
        if (typeof gridHelper !== "undefined" && gridHelper.material) {
            gridHelper.material.color.setHex(0xa855f7);
        }
    } else if (resolvedProfile === "HIGH_ALTITUDE_THIN_AIR") {
        scene.fog = new THREE.FogExp2(0x080c24, 0.020);
        particleMat.color.setHex(0xa5f3fc);
        particleMat.opacity = 0.50;
        particleMat.size = 0.05;
        ambientLight.color.setHex(0xc7d2fe);
        ambientLight.intensity = 0.95;
        keyLight.color.setHex(0xffffff);
        rimLight.color.setHex(0xf97316);

        // Stratospheric Cyan Altitude Grid
        if (typeof gridHelper !== "undefined" && gridHelper.material) {
            gridHelper.material.color.setHex(0x38bdf8);
        }
    } else { // ISA_STANDARD
        scene.fog = new THREE.FogExp2(0x070b14, 0.015);
        particleMat.color.setHex(0x00f0ff);
        particleMat.opacity = 0.35;
        particleMat.size = 0.05;
        ambientLight.color.setHex(0xffffff);
        ambientLight.intensity = 0.9;
        keyLight.color.setHex(0xffffff);
        rimLight.color.setHex(0xf97316);

        // Neon Cyan Digital Twin Test Bench Grid
        if (typeof gridHelper !== "undefined" && gridHelper.material) {
            gridHelper.material.color.setHex(0x38bdf8);
        }
    }
}

function applyClimateScenery(climateKey) {
    updateEnvironmentVisuals(climateKey);
}

window.applyClimateScenery = applyClimateScenery;
window.updateEnvironmentVisuals = updateEnvironmentVisuals;

// Parse URL query parameter "?climate=..." or "?env=..."
(function () {
    try {
        const urlParams = new URLSearchParams(window.location.search);
        const climateParam = urlParams.get("climate") || urlParams.get("env") || urlParams.get("profile");
        if (climateParam) {
            applyClimateScenery(climateParam);
        } else {
            applyClimateScenery("HIGH_ALTITUDE");
        }
    } catch (e) {}
})();

window.addEventListener("message", function (event) {
    if (event.data && (event.data.type === "SET_CLIMATE" || event.data.climate)) {
        applyClimateScenery(event.data.climate || event.data.profile);
    }
});

// ---------------------------------------------------------------------------
// 3. Realistic PBR Material Definitions (Matching Reference CAD Images)
// ---------------------------------------------------------------------------

// 3. Realistic PBR Material Definitions (Matching Reference CAD Images)
// ---------------------------------------------------------------------------

// 3. Realistic PBR Material Definitions (Matching Reference CAD Images)
// ---------------------------------------------------------------------------

// 1. Signature High-Gloss Crimson Red DOHC Valve Cover Material
const valveCoverRedMat = new THREE.MeshPhysicalMaterial({
    color: 0xd91c1c,
    metalness: 0.35,
    roughness: 0.15,
    clearcoat: 1.0,
    clearcoatRoughness: 0.05,
    envMapIntensity: 2.0,
    transparent: true,
    opacity: 0.35,
    depthWrite: true,
});

// 2. Cast Aluminum Engine Block & Head Material (Realistic dark matte cast metal)
const castAluminumMat = new THREE.MeshStandardMaterial({
    color: 0x5a6578,
    metalness: 0.45,
    roughness: 0.50,
    envMapIntensity: 0.8,
    transparent: true,
    opacity: 0.35,
    depthWrite: true,
});

// Dedicated Crystal-Clear Transparent Gearbox Housing Glass Material
const gearboxGlassMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    metalness: 0.1,
    roughness: 0.1,
    envMapIntensity: 1.5,
    transparent: true,
    opacity: 0.18,
    side: THREE.DoubleSide,
    depthWrite: false,
});

// 3. Machined Steel Crankshaft, Rods & Flywheel
const machinedSteelMat = new THREE.MeshStandardMaterial({
    color: 0x94a3b8,
    metalness: 0.95,
    roughness: 0.15,
    envMapIntensity: 1.8,
});

// 4. Polished Mirror Chrome (Pulleys, Flanges, Turbo Compressor Housing)
const chromeMat = new THREE.MeshStandardMaterial({
    color: 0xf8fafc,
    metalness: 0.98,
    roughness: 0.04,
    envMapIntensity: 2.5,
});

// 5. Dark Cast Iron Lower Block & Structural Frame (Dark matte slate, non-black)
const darkBlockMat = new THREE.MeshStandardMaterial({
    color: 0x3a4556,
    metalness: 0.40,
    roughness: 0.55,
    envMapIntensity: 0.7,
    transparent: true,
    opacity: 0.35,
    depthWrite: true,
});

// 6. Timing Chain Blue Guide Rail Material
const chainGuideBlueMat = new THREE.MeshStandardMaterial({
    color: 0x2563eb,
    metalness: 0.4,
    roughness: 0.4,
});

// 7. Metallic Slate Titanium / Satin Black Oil Pan Sump Material
const oilPanMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    metalness: 0.75,
    roughness: 0.35,
    envMapIntensity: 1.6,
    transparent: true,
    opacity: 0.88,
    depthWrite: true,
});

// 8. Iconic Glossy Blue Spin-On Oil Filter Material
const oilFilterBlueMat = new THREE.MeshPhysicalMaterial({
    color: 0x1d4ed8,
    metalness: 0.3,
    roughness: 0.15,
    clearcoat: 1.0,
    clearcoatRoughness: 0.08,
    envMapIntensity: 1.8,
});

// 9. 4-into-1 Swept Dark Titanium Exhaust Headers (EGT Thermal Glow Sync)
const exhaustHeaderMat = new THREE.MeshStandardMaterial({
    color: 0x272e3b,
    metalness: 0.82,
    roughness: 0.30,
    emissive: 0x000000,
    emissiveIntensity: 0.0,
    envMapIntensity: 1.2,
});

// 10. Cylinder Head Cooling Fins Material (CHT Thermal Sync)
const chtFinsMat = new THREE.MeshStandardMaterial({
    color: 0x00e676,
    metalness: 0.65,
    roughness: 0.3,
    emissive: 0x003311,
    emissiveIntensity: 0.2,
    envMapIntensity: 1.2,
    transparent: true,
    opacity: 0.35,
    depthWrite: true,
});

// Outer Engine Shell Materials Group for Transparency / Glass Cutaway Slider Controls
const shellMaterials = [castAluminumMat, valveCoverRedMat, oilPanMat, darkBlockMat, chtFinsMat];
let currentShellOpacity = 0.20;

function setEngineShellOpacity(val) {
    currentShellOpacity = parseFloat(val);
    shellMaterials.forEach(mat => {
        mat.transparent = currentShellOpacity < 0.99;
        mat.opacity = currentShellOpacity;
        mat.depthWrite = true;
        mat.needsUpdate = true;
    });
    const label = document.getElementById("opacity-val");
    if (label) label.innerText = Math.round(currentShellOpacity * 100) + "%";
    const slider = document.getElementById("opacity-slider");
    if (slider) slider.value = currentShellOpacity;
}

function toggleGlassCutaway() {
    if (currentShellOpacity > 0.5) {
        setEngineShellOpacity(0.35);
    } else {
        setEngineShellOpacity(1.0);
    }
}

window.setEngineShellOpacity = setEngineShellOpacity;
window.toggleGlassCutaway = toggleGlassCutaway;

// 11. Ribbed Synthetic Rubber Serpentine Belt & Hose Couplers Material
const serpentineBeltMat = new THREE.MeshStandardMaterial({
    color: 0x334155,
    metalness: 0.25,
    roughness: 0.65,
});

// 12. Carbon Composite Propeller Blade Material
const propBladeMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    metalness: 0.6,
    roughness: 0.30,
    envMapIntensity: 1.4,
});

// 13. Red High-Voltage Ignition Wires
const redWireMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    metalness: 0.3,
    roughness: 0.4,
});

// 14. Brass Details
const brassMat = new THREE.MeshStandardMaterial({
    color: 0xd4af37,
    metalness: 0.88,
    roughness: 0.22,
    envMapIntensity: 1.8,
});

// 15. Anodized Metallic Steel Output Drive Flange
const darkFlangeMat = new THREE.MeshStandardMaterial({
    color: 0x475569,
    metalness: 0.85,
    roughness: 0.25,
    envMapIntensity: 1.6,
});

// 16. Dark Anodized Steel Front Drive Pulleys Material
const darkPulleyMat = new THREE.MeshStandardMaterial({
    color: 0x222834,
    metalness: 0.85,
    roughness: 0.28,
    envMapIntensity: 1.2,
});

// 17. Intercooler & Charge Air Piping Materials
const chargePipeMat = new THREE.MeshStandardMaterial({
    color: 0x222834, metalness: 0.78, roughness: 0.32, envMapIntensity: 1.2
});
const siliconeCouplerMat = new THREE.MeshStandardMaterial({
    color: 0x1a56db, metalness: 0.0, roughness: 0.50
});
const intercoolerBodyMat = new THREE.MeshStandardMaterial({
    color: 0xb8c4cc, metalness: 0.80, roughness: 0.28
});
const intercoolerFinMat = new THREE.MeshStandardMaterial({
    color: 0x8fa0ae, metalness: 0.88, roughness: 0.20
});

// ---------------------------------------------------------------------------
// 4. Advanced Geometric Helpers
// ---------------------------------------------------------------------------
function createZCylinder(radiusTop, radiusBottom, height, radialSegments, material) {
    const geo = new THREE.CylinderGeometry(radiusTop, radiusBottom, height, radialSegments);
    geo.rotateX(Math.PI / 2);
    return new THREE.Mesh(geo, material);
}

function createXCylinder(radiusTop, radiusBottom, height, radialSegments, material) {
    const geo = new THREE.CylinderGeometry(radiusTop, radiusBottom, height, radialSegments);
    geo.rotateZ(Math.PI / 2);
    return new THREE.Mesh(geo, material);
}

function createZCone(radius, height, radialSegments, material) {
    const geo = new THREE.ConeGeometry(radius, height, radialSegments);
    geo.rotateX(Math.PI / 2);
    return new THREE.Mesh(geo, material);
}

// V-Belt Pulley Lathe Profile Generator
function createVPulleyMesh(radius, thickness, material) {
    const points = [];
    const rIn = radius * 0.82;
    const rOut = radius;
    const h = thickness / 2;

    points.push(new THREE.Vector2(rIn, -h));
    points.push(new THREE.Vector2(rOut, -h));
    points.push(new THREE.Vector2(rOut - 0.03, -h * 0.4));
    points.push(new THREE.Vector2(rOut - 0.03, h * 0.4));
    points.push(new THREE.Vector2(rOut, h));
    points.push(new THREE.Vector2(rIn, h));

    const geo = new THREE.LatheGeometry(points, 32);
    geo.rotateX(Math.PI / 2);
    return new THREE.Mesh(geo, material);
}

// 36-Teeth Extruded Cam Sprockets Generator
function createExtrudedCamGearMesh(radius, thickness, numTeeth, material) {
    const shape = new THREE.Shape();
    const outerR = radius;
    const innerR = radius - 0.035;

    for (let i = 0; i < numTeeth; i++) {
        const a1 = (i * 2 * Math.PI) / numTeeth;
        const a2 = a1 + (Math.PI / numTeeth) * 0.45;
        const a3 = a1 + (Math.PI / numTeeth) * 0.85;
        const a4 = ((i + 1) * 2 * Math.PI) / numTeeth;

        if (i === 0) {
            shape.moveTo(Math.cos(a1) * innerR, Math.sin(a1) * innerR);
        }
        shape.lineTo(Math.cos(a1) * outerR, Math.sin(a1) * outerR);
        shape.lineTo(Math.cos(a2) * outerR, Math.sin(a2) * outerR);
        shape.lineTo(Math.cos(a3) * innerR, Math.sin(a3) * innerR);
        shape.lineTo(Math.cos(a4) * innerR, Math.sin(a4) * innerR);
    }

    const holePath = new THREE.Path();
    holePath.absarc(0, 0, radius * 0.4, 0, Math.PI * 2, true);
    shape.holes.push(holePath);

    const extrudeSettings = {
        depth: thickness,
        bevelEnabled: true,
        bevelSegments: 3,
        steps: 1,
        bevelSize: 0.006,
        bevelThickness: 0.006,
    };

    const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    return new THREE.Mesh(geo, material);
}

// Eccentric Teardrop Cam Lobe Generator for DOHC Camshafts
function createCamLobeMesh(material) {
    const shape = new THREE.Shape();
    const rBase = 0.038;
    const rTip = 0.016;
    const hLobe = 0.072;

    shape.absarc(0, 0, rBase, Math.PI / 4, (Math.PI * 3) / 4, false);
    shape.lineTo(-rTip, hLobe);
    shape.absarc(0, hLobe, rTip, Math.PI, 0, true);
    shape.lineTo(rBase * Math.cos(Math.PI / 4), rBase * Math.sin(Math.PI / 4));

    const extrudeSettings = {
        depth: 0.032,
        bevelEnabled: true,
        bevelSegments: 3,
        steps: 1,
        bevelSize: 0.004,
        bevelThickness: 0.004,
    };

    const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    geo.center();
    return new THREE.Mesh(geo, material);
}

// Organic Beveled DOHC Valve Cover Extrusion Generator
function createOrganicValveCoverMesh(width, height, depth, material) {
    const shape = new THREE.Shape();
    const w = width / 2;
    const h = height;
    const r = 0.08;

    shape.moveTo(-w + r, 0);
    shape.lineTo(w - r, 0);
    shape.quadraticCurveTo(w, 0, w, r);
    shape.lineTo(w * 0.85, h - r);
    shape.quadraticCurveTo(w * 0.82, h, w * 0.82 - r, h);
    shape.lineTo(-w * 0.82 + r, h);
    shape.quadraticCurveTo(-w * 0.82, h, -w * 0.85, h - r);
    shape.lineTo(-w, r);
    shape.quadraticCurveTo(-w, 0, -w + r, 0);

    const extrudeSettings = {
        depth: depth,
        bevelEnabled: true,
        bevelSegments: 8,
        steps: 2,
        bevelSize: 0.04,
        bevelThickness: 0.04,
    };

    const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    geo.center();
    return new THREE.Mesh(geo, material);
}

// ---------------------------------------------------------------------------
// 5. Build High-Fidelity Inline-4 Aero Engine Master Assembly
// ---------------------------------------------------------------------------
const engineMasterGroup = new THREE.Group();
engineMasterGroup.position.set(0, 0, 0);
scene.add(engineMasterGroup);

// Global Engine Layout Constants
const cylinderSpacing = 0.32;
const startZ = 0.48;

// ---------------------------------------------------------------------------
// A. REALISTIC CRANKCASE STRUCTURE (Scalloped Cylinder Bays, Main Caps & Side Ribs)
// ---------------------------------------------------------------------------
const crankcaseGroup = new THREE.Group();
crankcaseGroup.name = "OIL_SUMP_FILTER";
engineMasterGroup.add(crankcaseGroup);

// 1. Four Scalloped Crankcase Bays (Semi-Cylindrical Lower Troughs for Crank Throws)
for (let i = 0; i < 4; i++) {
    const zPos = startZ - i * cylinderSpacing;

    // Semi-cylindrical lower crank arch trough
    const troughGeo = new THREE.CylinderGeometry(0.34, 0.34, 0.28, 24, 1, false, Math.PI, Math.PI);
    troughGeo.rotateZ(Math.PI); // Arc points downward
    troughGeo.rotateX(Math.PI / 2);
    const troughMesh = new THREE.Mesh(troughGeo, castAluminumMat);
    troughMesh.position.set(0, 0.52, zPos);
    troughMesh.castShadow = true;
    crankcaseGroup.add(troughMesh);

    // Contoured Side Skirt Saddles
    const saddleL = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.24, 0.28), castAluminumMat);
    saddleL.position.set(-0.32, 0.54, zPos);
    crankcaseGroup.add(saddleL);

    const saddleR = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.24, 0.28), castAluminumMat);
    saddleR.position.set(0.32, 0.54, zPos);
    crankcaseGroup.add(saddleR);
}

// 2. Continuous Upper Skirt Rail Flanges & Longitudinal Stiffeners
const skirtRailL = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.26, 1.42), castAluminumMat);
skirtRailL.position.set(-0.32, 0.55, 0);
crankcaseGroup.add(skirtRailL);

const skirtRailR = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.26, 1.42), castAluminumMat);
skirtRailR.position.set(0.32, 0.55, 0);
crankcaseGroup.add(skirtRailR);

// 3. Five Heavy Internal Main Bearing Bulkheads & Bearing Caps
for (let bz of [-0.64, -0.32, 0.0, 0.32, 0.64]) {
    // Main bearing transverse bulkhead web
    const bulkhead = createZCylinder(0.34, 0.34, 0.045, 24, castAluminumMat);
    bulkhead.position.set(0, 0.52, bz);
    crankcaseGroup.add(bulkhead);

    // Machined Main Bearing Cap (Lower Arch)
    const capGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.05, 20, 1, false, Math.PI, Math.PI);
    capGeo.rotateZ(Math.PI);
    capGeo.rotateX(Math.PI / 2);
    const mainCap = new THREE.Mesh(capGeo, darkBlockMat);
    mainCap.position.set(0, 0.44, bz);
    crankcaseGroup.add(mainCap);

    // Main Bearing Cap Fastener Studs & Nuts
    for (let cx of [-0.12, 0.12]) {
        const capNut = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.04, 6), chromeMat);
        capNut.position.set(cx, 0.42, bz);
        crankcaseGroup.add(capNut);
    }

    // Side Cross-Bolting Bosses (Real Racing Engine Crankcase Cross-Bolts!)
    for (let sideX of [-0.34, 0.34]) {
        const boss = createXCylinder(0.025, 0.025, 0.05, 12, castAluminumMat);
        boss.position.set(sideX, 0.52, bz);
        crankcaseGroup.add(boss);

        const crossBolt = createXCylinder(0.012, 0.012, 0.07, 6, chromeMat);
        crossBolt.position.set(sideX, 0.52, bz);
        crankcaseGroup.add(crossBolt);
    }
}

// 4. Outer Crankcase Stiffening Ribbing Grid
for (let rz of [-0.48, -0.16, 0.16, 0.48]) {
    const diagRibL = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.22, 0.03), castAluminumMat);
    diagRibL.rotation.z = Math.PI / 6;
    diagRibL.position.set(-0.33, 0.54, rz);
    crankcaseGroup.add(diagRibL);

    const diagRibR = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.22, 0.03), castAluminumMat);
    diagRibR.rotation.z = -Math.PI / 6;
    diagRibR.position.set(0.33, 0.54, rz);
    crankcaseGroup.add(diagRibR);
}

// 5. Front Nose Seal Housing & Rear Main Seal Housing
const frontNoseSeal = createZCylinder(0.20, 0.20, 0.06, 24, castAluminumMat);
frontNoseSeal.position.set(0, 0.52, 0.70);
crankcaseGroup.add(frontNoseSeal);

const rearMainSeal = createZCylinder(0.22, 0.22, 0.05, 24, darkBlockMat);
rearMainSeal.position.set(0, 0.52, -0.70);
crankcaseGroup.add(rearMainSeal);

// ---------------------------------------------------------------------------
// A2. REALISTIC 3D FORGED STEEL CRANKSHAFT ASSEMBLY (4-Cylinder Offset Throws)
// ---------------------------------------------------------------------------
const crankshaftGroup = new THREE.Group();
crankshaftGroup.name = "CRANKSHAFT_ASSEMBLY";
crankshaftGroup.position.set(0, 0.52, 0); // Central shaft axis at Y = 0.52
engineMasterGroup.add(crankshaftGroup);

// Central Main Bearing Journals along Z-axis (5 Main Bearings at z = -0.64, -0.32, 0.0, 0.32, 0.64)
const mainJournalZCoords = [-0.64, -0.32, 0.0, 0.32, 0.64];
for (let bz of mainJournalZCoords) {
    const mainJournal = createZCylinder(0.075, 0.075, 0.05, 24, machinedSteelMat);
    mainJournal.position.set(0, 0, bz);
    mainJournal.castShadow = true;
    crankshaftGroup.add(mainJournal);
}

// 4 Offset Crankpins & 8 Wedge Counterweight Webs (1-3-4-2 throw arrangement)
const strokeRadiusVal = 0.095;
for (let i = 0; i < 4; i++) {
    const zPos = startZ - i * cylinderSpacing;
    const crankPhase = (i === 0 || i === 3) ? 0 : Math.PI;
    
    // Offset Crankpin Journal (Big-end connecting rod mounting journal)
    const pinX = Math.cos(crankPhase) * strokeRadiusVal;
    const pinY = Math.sin(crankPhase) * strokeRadiusVal;
    
    const crankPin = createZCylinder(0.065, 0.065, 0.06, 24, machinedSteelMat);
    crankPin.position.set(pinX, pinY, zPos);
    crankPin.castShadow = true;
    crankshaftGroup.add(crankPin);

    // Flanking Heavy Wedge Counterweights (Balanced opposite to crankpin)
    for (let sideZ of [zPos + 0.042, zPos - 0.042]) {
        const webGroup = new THREE.Group();
        webGroup.position.set(0, 0, sideZ);
        webGroup.rotation.z = crankPhase + Math.PI; // Points opposite to crankpin!
        crankshaftGroup.add(webGroup);

        const cwShape = new THREE.Shape();
        cwShape.moveTo(-0.14, 0);
        cwShape.quadraticCurveTo(-0.17, 0.19, 0, 0.21);
        cwShape.quadraticCurveTo(0.17, 0.19, 0.14, 0);
        cwShape.lineTo(-0.14, 0);

        const cwExtrude = new THREE.ExtrudeGeometry(cwShape, {
            depth: 0.024,
            bevelEnabled: true,
            bevelSegments: 3,
            bevelSize: 0.005,
            bevelThickness: 0.005
        });
        cwExtrude.center();
        const cwMesh = new THREE.Mesh(cwExtrude, machinedSteelMat);
        cwMesh.castShadow = true;
        webGroup.add(cwMesh);
    }
}

// ---------------------------------------------------------------------------
// B. REALISTIC ORGANIC STEPPED WET SUMP OIL PAN (Gapless Sealed Flange & Fins)
// ---------------------------------------------------------------------------
const oilPanGroup = new THREE.Group();
crankcaseGroup.add(oilPanGroup);

// 1. Continuous Outer Perimeter Flange Rail (Flush 0-Gap Seal against Crankcase Skirt y=0.42-0.45)
const panRailFlange = new THREE.Mesh(
    new THREE.BoxGeometry(0.72, 0.04, 1.44),
    oilPanMat
);
panRailFlange.position.set(0, 0.425, 0);
panRailFlange.castShadow = true;
oilPanGroup.add(panRailFlange);

// Flange Gasket Sealing Ridge / Inner Rim Lip
const flangeSealingLip = new THREE.Mesh(
    new THREE.BoxGeometry(0.68, 0.025, 1.40),
    darkBlockMat
);
flangeSealingLip.position.set(0, 0.44, 0);
oilPanGroup.add(flangeSealingLip);

// 22 Machined Chrome Perimeter Mounting Flange Bolts with Copper Crush Washers
const boltPositionsZ = [-0.66, -0.44, -0.22, 0.0, 0.22, 0.44, 0.66];
for (let bz of boltPositionsZ) {
    for (let bx of [-0.33, 0.33]) {
        const washer = createZCylinder(0.020, 0.020, 0.008, 12, brassMat);
        washer.position.set(bx, 0.442, bz);
        oilPanGroup.add(washer);

        const bolt = createZCylinder(0.014, 0.014, 0.032, 6, chromeMat);
        bolt.position.set(bx, 0.455, bz);
        oilPanGroup.add(bolt);
    }
}
for (let bx of [-0.22, -0.11, 0.11, 0.22]) {
    for (let bz of [-0.67, 0.67]) {
        const washer = createZCylinder(0.020, 0.020, 0.008, 12, brassMat);
        washer.position.set(bx, 0.442, bz);
        oilPanGroup.add(washer);

        const bolt = createZCylinder(0.014, 0.014, 0.032, 6, chromeMat);
        bolt.position.set(bx, 0.455, bz);
        oilPanGroup.add(bolt);
    }
}

// 2. Upper Skirt Transition Body (Completely Seals Gap to Engine Crankcase Skirt)
const upperPanSkirt = new THREE.Mesh(
    new THREE.BoxGeometry(0.66, 0.08, 1.40),
    oilPanMat
);
upperPanSkirt.position.set(0, 0.38, 0);
upperPanSkirt.castShadow = true;
oilPanGroup.add(upperPanSkirt);

// 3. Shallow Front Clearance Basin (Front Sump Clearance for Cross-Member)
const shallowPanFront = new THREE.Mesh(
    new THREE.BoxGeometry(0.62, 0.12, 0.46),
    oilPanMat
);
shallowPanFront.position.set(0, 0.31, 0.45);
shallowPanFront.castShadow = true;
oilPanGroup.add(shallowPanFront);

// Angled Front Nose Bevel
const frontBevel = new THREE.Mesh(
    new THREE.BoxGeometry(0.60, 0.10, 0.14),
    oilPanMat
);
frontBevel.rotation.x = Math.PI / 6;
frontBevel.position.set(0, 0.33, 0.66);
oilPanGroup.add(frontBevel);

// 4. Smooth Angled Transition Ramp Connecting Shallow Front to Deep Rear Sump
const transitionRamp = new THREE.Mesh(
    new THREE.BoxGeometry(0.61, 0.04, 0.34),
    oilPanMat
);
transitionRamp.rotation.x = Math.PI / 6; // 30-degree smooth sloped ramp wall
transitionRamp.position.set(0, 0.24, 0.12);
transitionRamp.castShadow = true;
oilPanGroup.add(transitionRamp);

// Triangular Side-Wall Wedges for Transition Ramp
for (let sideX of [-0.30, 0.30]) {
    const sideWall = new THREE.Mesh(
        new THREE.BoxGeometry(0.04, 0.16, 0.30),
        oilPanMat
    );
    sideWall.position.set(sideX, 0.23, 0.12);
    oilPanGroup.add(sideWall);
}

// Longitudinal Stiffener Ribs Under Ramp (Matching Reference CAD Drawing)
for (let rx of [-0.18, 0.0, 0.18]) {
    const rampRib = new THREE.Mesh(
        new THREE.BoxGeometry(0.02, 0.03, 0.32),
        oilPanMat
    );
    rampRib.rotation.x = Math.PI / 6;
    rampRib.position.set(rx, 0.22, 0.12);
    oilPanGroup.add(rampRib);
}

// 5. Deep Rear Wet-Sump Collector Reservoir (Oil Well under Cyl 1 & 2)
const deepSumpReservoir = new THREE.Mesh(
    new THREE.BoxGeometry(0.61, 0.24, 0.68),
    oilPanMat
);
deepSumpReservoir.position.set(0, 0.15, -0.32);
deepSumpReservoir.castShadow = true;
oilPanGroup.add(deepSumpReservoir);

// Tapered Bottom Base Floor of Deep Sump
const deepSumpBase = new THREE.Mesh(
    new THREE.BoxGeometry(0.56, 0.04, 0.62),
    oilPanMat
);
deepSumpBase.position.set(0, 0.02, -0.32);
oilPanGroup.add(deepSumpBase);

// 6. External Horizontal Cooling Fins (Lower Deep Sump Cooling Fins)
for (let r = 0; r < 6; r++) {
    const fin = new THREE.Mesh(
        new THREE.BoxGeometry(0.63, 0.016, 0.66),
        oilPanMat
    );
    fin.position.set(0, 0.04 + r * 0.03, -0.32);
    oilPanGroup.add(fin);
}

// Vertical Reinforcing Stiffener Gussets on Pan Side Walls (Matching Real Stamped/Cast Pan)
for (let bz of [-0.58, -0.45, -0.32, -0.19, -0.06]) {
    for (let sideX of [-0.312, 0.312]) {
        const gusset = new THREE.Mesh(
            new THREE.BoxGeometry(0.018, 0.20, 0.035),
            oilPanMat
        );
        gusset.position.set(sideX, 0.15, bz);
        oilPanGroup.add(gusset);
    }
}

// 7. Hex Oil Drain Plug Assembly (Angled Boss, Copper Washer, Hex Bolt at Bottom Rear Right)
const drainBossGroup = new THREE.Group();
drainBossGroup.position.set(0.24, 0.05, -0.60);
drainBossGroup.rotation.z = -Math.PI / 4; // 45 degree outward drain boss
oilPanGroup.add(drainBossGroup);

const drainPlugBoss = new THREE.Mesh(
    new THREE.CylinderGeometry(0.042, 0.042, 0.03, 16),
    darkBlockMat
);
drainBossGroup.add(drainPlugBoss);

const drainCopperWasher = new THREE.Mesh(
    new THREE.CylinderGeometry(0.038, 0.038, 0.008, 16),
    brassMat
);
drainCopperWasher.position.set(0, -0.018, 0);
drainBossGroup.add(drainCopperWasher);

const drainPlugHex = new THREE.Mesh(
    new THREE.CylinderGeometry(0.030, 0.030, 0.035, 6),
    chromeMat
);
drainPlugHex.position.set(0, -0.036, 0);
drainBossGroup.add(drainPlugHex);

// 8. Internal Oil Pickup Tube & Strainer Mesh Foot
const pickupPipePoints = [
    new THREE.Vector3(0, 0.38, 0),
    new THREE.Vector3(0, 0.20, -0.15),
    new THREE.Vector3(0, 0.08, -0.32)
];
const pickupPath = new THREE.CatmullRomCurve3(pickupPipePoints);
const pickupGeo = new THREE.TubeGeometry(pickupPath, 16, 0.022, 12, false);
const pickupPipe = new THREE.Mesh(pickupGeo, brassMat);
oilPanGroup.add(pickupPipe);

const pickupScreenFoot = new THREE.Mesh(
    new THREE.CylinderGeometry(0.065, 0.065, 0.02, 16),
    brassMat
);
pickupScreenFoot.position.set(0, 0.06, -0.32);
oilPanGroup.add(pickupScreenFoot);

// Internal Anti-Slosh Surge Baffle Plate
const bafflePlate = new THREE.Mesh(
    new THREE.BoxGeometry(0.58, 0.008, 0.52),
    darkBlockMat
);
bafflePlate.position.set(0, 0.28, -0.28);
oilPanGroup.add(bafflePlate);

// Stainless Steel Dipstick Tube & Bracket
const dipstickTubePoints = [
    new THREE.Vector3(-0.33, 0.32, -0.25),
    new THREE.Vector3(-0.35, 0.75, -0.18),
    new THREE.Vector3(-0.34, 1.25, -0.12)
];
const dipstickPath = new THREE.CatmullRomCurve3(dipstickTubePoints);
const dipstickGeo = new THREE.TubeGeometry(dipstickPath, 20, 0.012, 12, false);
const dipstickTube = new THREE.Mesh(dipstickGeo, chromeMat);
crankcaseGroup.add(dipstickTube);

// Yellow Dipstick Handle Ring at Top
const dipstickHandle = new THREE.Mesh(
    new THREE.TorusGeometry(0.025, 0.006, 12, 24),
    new THREE.MeshStandardMaterial({ color: 0xeab308, roughness: 0.3 })
);
dipstickHandle.position.set(-0.34, 1.27, -0.12);
crankcaseGroup.add(dipstickHandle);

// Iconic Glossy Blue Spin-On Oil Filter
const oilFilterGroup = new THREE.Group();
oilFilterGroup.position.set(0.38, 0.36, 0.35);
oilFilterGroup.rotation.z = -Math.PI / 4;
crankcaseGroup.add(oilFilterGroup);

const oilFilterMount = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.07, 0.05, 24),
    castAluminumMat
);
oilFilterGroup.add(oilFilterMount);

const oilFilterCanister = new THREE.Mesh(
    new THREE.CylinderGeometry(0.075, 0.075, 0.22, 32),
    oilFilterBlueMat
);
oilFilterCanister.position.y = -0.12;
oilFilterCanister.castShadow = true;
oilFilterGroup.add(oilFilterCanister);

// ---------------------------------------------------------------------------
// C. Cast Aluminum Inline-4 Cylinder Block & Fins
// ---------------------------------------------------------------------------
const cylinderBlockGroup = new THREE.Group();
cylinderBlockGroup.name = "CYLINDER_HEAD";
engineMasterGroup.add(cylinderBlockGroup);

const pistons = [];
const conRods = [];
const combustionChambers = [];
const cylinderValves = [];
const valvesByCylinder = [[], [], [], []];
const sparkArcs = [];

// Firing Order 1-3-4-2 Crank Phase Offsets (Cyl 1: 0, Cyl 2: 3*PI, Cyl 3: PI, Cyl 4: 2*PI)
const firingPhaseOffsets = [0, 3 * Math.PI, Math.PI, 2 * Math.PI];

// 1. Four Contoured Cylindrical Outer Barrels
for (let i = 0; i < 4; i++) {
    const zPos = startZ - i * cylinderSpacing;

    const barrelHump = new THREE.Mesh(
        new THREE.CylinderGeometry(0.26, 0.27, 0.44, 24),
        castAluminumMat
    );
    barrelHump.position.set(0, 0.92, zPos);
    barrelHump.castShadow = true;
    cylinderBlockGroup.add(barrelHump);
}

// 2. Machined Upper Cylinder Head Deck Plate Base
const deckPlate = new THREE.Mesh(
    new THREE.BoxGeometry(0.60, 0.04, 1.40),
    castAluminumMat
);
deckPlate.position.set(0, 1.15, 0);
cylinderBlockGroup.add(deckPlate);

// 3. Outward Flared Lower Skirt Rail
const skirtRail = new THREE.Mesh(
    new THREE.BoxGeometry(0.66, 0.08, 1.40),
    castAluminumMat
);
skirtRail.position.set(0, 0.72, 0);
cylinderBlockGroup.add(skirtRail);

// 4. Side Structural Reinforcement Ribbing Webs
for (let ribZ of [-0.48, -0.16, 0.16, 0.48]) {
    const sideRibL = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.42, 0.03), castAluminumMat);
    sideRibL.position.set(-0.30, 0.92, ribZ);
    cylinderBlockGroup.add(sideRibL);

    const sideRibR = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.42, 0.03), castAluminumMat);
    sideRibR.position.set(0.30, 0.92, ribZ);
    cylinderBlockGroup.add(sideRibR);
}

// 5. Four Cylinder Bores, Pistons, DOHC Valves, Spark Plugs & Kinematics
const valveSpringMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.9, roughness: 0.2 });
const sparkCeramicMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.15 });

for (let i = 0; i < 4; i++) {
    const zPos = startZ - i * cylinderSpacing;

    // Inner Cylinder Sleeve Liner Window
    const sleeve = new THREE.Mesh(
        new THREE.CylinderGeometry(0.18, 0.18, 0.44, 32),
        darkBlockMat
    );
    sleeve.position.set(0, 0.92, zPos);
    cylinderBlockGroup.add(sleeve);

    // Chrome Deck Chamfer Ring
    const deckRing = new THREE.TorusGeometry(0.18, 0.01, 16, 32);
    const deckRingMesh = new THREE.Mesh(deckRing, chromeMat);
    deckRingMesh.rotation.x = Math.PI / 2;
    deckRingMesh.position.set(0, 1.15, zPos);
    cylinderBlockGroup.add(deckRingMesh);

    // Exterior Cooling Fins (CHT Thermal Sync)
    for (let f = 0; f < 6; f++) {
        const fin = new THREE.Mesh(
            new THREE.BoxGeometry(0.72, 0.016, 0.26),
            chtFinsMat
        );
        fin.position.set(0, 0.74 + f * 0.065, zPos);
        cylinderBlockGroup.add(fin);
    }

    // High-Detail Piston Crown Assembly
    const pistonGroup = new THREE.Group();
    pistonGroup.position.set(0, 0.92, zPos);
    cylinderBlockGroup.add(pistonGroup);

    const pistonCrown = new THREE.Mesh(
        new THREE.CylinderGeometry(0.175, 0.175, 0.16, 32),
        machinedSteelMat
    );
    pistonCrown.castShadow = true;
    pistonGroup.add(pistonCrown);

    // Dished Piston Top Recess
    const dishRecess = new THREE.Mesh(
        new THREE.CylinderGeometry(0.14, 0.14, 0.02, 32),
        darkBlockMat
    );
    dishRecess.position.y = 0.075;
    pistonGroup.add(dishRecess);

    // 3 Piston Ring Grooves
    for (let r of [0.04, 0.01, -0.02]) {
        const ringTorus = new THREE.TorusGeometry(0.176, 0.005, 12, 32);
        const ringMesh = new THREE.Mesh(ringTorus, darkBlockMat);
        ringMesh.rotation.x = Math.PI / 2;
        ringMesh.position.y = r;
        pistonGroup.add(ringMesh);
    }

    // Stainless Wrist Pin Shaft
    const wristPin = createXCylinder(0.03, 0.03, 0.28, 16, machinedSteelMat);
    wristPin.position.set(0, -0.02, 0);
    pistonGroup.add(wristPin);

    // Realistic I-Beam Connecting Rod Assembly
    const conRodGroup = new THREE.Group();
    conRodGroup.position.set(0, 0.74, zPos);
    cylinderBlockGroup.add(conRodGroup);

    const rodShank = new THREE.Mesh(
        new THREE.BoxGeometry(0.045, 0.32, 0.05),
        castAluminumMat
    );
    conRodGroup.add(rodShank);

    // Big-End Crank Journal Eye & Split Line Bolts
    const bigEnd = createZCylinder(0.07, 0.07, 0.055, 24, castAluminumMat);
    bigEnd.position.y = -0.16;
    conRodGroup.add(bigEnd);

    for (let bx of [-0.05, 0.05]) {
        const bolt = createZCylinder(0.01, 0.01, 0.065, 6, chromeMat);
        bolt.position.set(bx, -0.16, 0);
        conRodGroup.add(bolt);
    }

    // Small-End Wrist Pin Eye & Bronze Bushing
    const smallEnd = createZCylinder(0.045, 0.045, 0.05, 24, castAluminumMat);
    smallEnd.position.y = 0.16;
    conRodGroup.add(smallEnd);

    const bushing = createZCylinder(0.032, 0.032, 0.052, 24, brassMat);
    bushing.position.y = 0.16;
    conRodGroup.add(bushing);

    // Inline-4 Firing Order Phase Offsets (1-3-4-2: 0, PI, PI, 0 for piston mechanical travel)
    const phaseOffset = (i === 0 || i === 3) ? 0 : Math.PI;
    pistons.push({ mesh: pistonGroup, phase: phaseOffset, baseZ: zPos });
    conRods.push({ mesh: conRodGroup, phase: phaseOffset, baseZ: zPos });

    // -----------------------------------------------------------------------
    // DOHC POPPET VALVES (4 Valves per Cylinder: 2 Intake, 2 Exhaust)
    // -----------------------------------------------------------------------
    const valveOffsets = [
        { type: "INTAKE",  x: -0.09, zOffset: 0.05 },
        { type: "INTAKE",  x: -0.09, zOffset: -0.05 },
        { type: "EXHAUST", x:  0.09, zOffset: 0.05 },
        { type: "EXHAUST", x:  0.09, zOffset: -0.05 }
    ];

    valveOffsets.forEach(vConfig => {
        const vGroup = new THREE.Group();
        vGroup.position.set(vConfig.x, 1.22, zPos + vConfig.zOffset);
        cylinderBlockGroup.add(vGroup);

        // Valve Stem (Stainless polished rod)
        const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.16, 12), machinedSteelMat);
        vGroup.add(stem);

        // Valve Head Beveled Disk (Poppet face seating in cylinder dome)
        const head = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.045, 0.015, 24), chromeMat);
        head.position.y = -0.08;
        vGroup.add(head);

        // Spiral Valve Spring Coil
        for (let sp = 0; sp < 5; sp++) {
            const coilRing = new THREE.Mesh(new THREE.TorusGeometry(0.018, 0.004, 8, 16), valveSpringMat);
            coilRing.rotation.x = Math.PI / 2;
            coilRing.position.y = -0.02 + sp * 0.018;
            vGroup.add(coilRing);
        }

        // Valve Spring Retainer Top Cap
        const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.01, 16), darkBlockMat);
        cap.position.y = 0.075;
        vGroup.add(cap);

        const vObj = {
            mesh: vGroup,
            type: vConfig.type,
            cylIndex: i,
            baseY: 1.22
        };
        cylinderValves.push(vObj);
        if (valvesByCylinder[i]) valvesByCylinder[i].push(vObj);
    });

    // -----------------------------------------------------------------------
    // SPARK PLUG & HIGH-VOLTAGE IGNITION ARC FLASH ASSEMBLY
    // -----------------------------------------------------------------------
    const sparkGroup = new THREE.Group();
    sparkGroup.position.set(0, 1.24, zPos);
    cylinderBlockGroup.add(sparkGroup);

    // Ceramic Insulator & Threaded Hex Body
    const ceramic = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.12, 16), sparkCeramicMat);
    sparkGroup.add(ceramic);

    const hexBody = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.04, 6), machinedSteelMat);
    hexBody.position.y = -0.02;
    sparkGroup.add(hexBody);

    const electrodePin = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.03, 8), chromeMat);
    electrodePin.position.y = -0.075;
    sparkGroup.add(electrodePin);

    // High-Voltage Electric Blue/White Ignition Arc Flash Sphere
    const arcGeo = new THREE.SphereGeometry(0.035, 16, 16);
    const arcMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.95 });
    const arcMesh = new THREE.Mesh(arcGeo, arcMat);
    arcMesh.position.y = -0.09;
    arcMesh.visible = false;
    sparkGroup.add(arcMesh);

    const coreGeo = new THREE.SphereGeometry(0.018, 12, 12);
    const coreMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    coreMesh.position.y = -0.09;
    coreMesh.visible = false;
    sparkGroup.add(coreMesh);

    sparkArcs.push({ arcMesh: arcMesh, coreMesh: coreMesh, cylIndex: i });

    // -----------------------------------------------------------------------
    // DUAL-LAYER VOLUMETRIC COMBUSTION CHAMBER (Flame Expansion + White-Hot Core)
    // -----------------------------------------------------------------------
    const chamberMat = new THREE.MeshStandardMaterial({
        color: 0x00d2ff,
        emissive: 0x00a2ff,
        emissiveIntensity: 1.5,
        transparent: true,
        opacity: 0.65,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    const chamberGeo = new THREE.CylinderGeometry(0.172, 0.172, 0.20, 24);
    const chamberMesh = new THREE.Mesh(chamberGeo, chamberMat);
    chamberMesh.position.set(0, 1.05, zPos);
    cylinderBlockGroup.add(chamberMesh);

    const flameCoreMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: 0xff4400,
        emissiveIntensity: 4.5,
        transparent: true,
        opacity: 0.85,
        depthWrite: false
    });
    const flameCoreGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.16, 20);
    const flameCoreMesh = new THREE.Mesh(flameCoreGeo, flameCoreMat);
    flameCoreMesh.position.set(0, 1.05, zPos);
    flameCoreMesh.visible = false;
    cylinderBlockGroup.add(flameCoreMesh);

    combustionChambers.push({
        mesh: chamberMesh,
        mat: chamberMat,
        coreMesh: flameCoreMesh,
        coreMat: flameCoreMat,
        cylIndex: i,
        baseZ: zPos
    });
}

// ---------------------------------------------------------------------------
// C2. DOHC TWIN CAMSHAFTS, CAM LOBES & ROCKER ARM SHAFTS ASSEMBLY
// ---------------------------------------------------------------------------
const dohcValveTrainGroup = new THREE.Group();
dohcValveTrainGroup.name = "DOHC_CAMSHAFTS_ROCKERS";
cylinderBlockGroup.add(dohcValveTrainGroup);

// Twin Camshaft Groups
const intakeCamshaftGroup = new THREE.Group();
intakeCamshaftGroup.position.set(-0.15, 1.38, 0);
dohcValveTrainGroup.add(intakeCamshaftGroup);

const intakeCamShaftBar = createZCylinder(0.024, 0.024, 1.38, 24, machinedSteelMat);
intakeCamshaftGroup.add(intakeCamShaftBar);

const exhaustCamshaftGroup = new THREE.Group();
exhaustCamshaftGroup.position.set(0.15, 1.38, 0);
dohcValveTrainGroup.add(exhaustCamshaftGroup);

const exhaustCamShaftBar = createZCylinder(0.024, 0.024, 1.38, 24, machinedSteelMat);
exhaustCamshaftGroup.add(exhaustCamShaftBar);

// 5 Main Camshaft Journal Bearing Caps
for (let bz of [-0.64, -0.32, 0.0, 0.32, 0.64]) {
    for (let camX of [-0.15, 0.15]) {
        const camCap = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.032, 0.045), castAluminumMat);
        camCap.position.set(camX, 1.40, bz);
        dohcValveTrainGroup.add(camCap);

        for (let bx of [-0.024, 0.024]) {
            const bolt = createZCylinder(0.006, 0.006, 0.038, 6, chromeMat);
            bolt.position.set(camX + bx, 1.40, bz);
            dohcValveTrainGroup.add(bolt);
        }
    }
}

// 2 Longitudinal Rocker Arm Pivot Shafts
const intakeRockerShaft = createZCylinder(0.016, 0.016, 1.38, 20, machinedSteelMat);
intakeRockerShaft.position.set(-0.09, 1.32, 0);
dohcValveTrainGroup.add(intakeRockerShaft);

const exhaustRockerShaft = createZCylinder(0.016, 0.016, 1.38, 20, machinedSteelMat);
exhaustRockerShaft.position.set(0.09, 1.32, 0);
dohcValveTrainGroup.add(exhaustRockerShaft);

// 16 Forged Steel Rocker Arms & 16 Eccentric Cam Lobes (Phased per cylinder 1-3-4-2)
const rockerArms = [];

for (let i = 0; i < 4; i++) {
    const zPos = startZ - i * cylinderSpacing;
    const fireOffset = firingPhaseOffsets[i] || 0;
    const camLobePhase = fireOffset * 0.5;

    // Intake Cam Lobes & Rocker Arms (2 per cylinder)
    for (let zOff of [0.05, -0.05]) {
        const vz = zPos + zOff;

        // 1. Intake Cam Lobe on Intake Camshaft
        const inLobe = createCamLobeMesh(machinedSteelMat);
        inLobe.position.set(0, 0, vz);
        inLobe.rotation.z = camLobePhase + Math.PI / 4;
        intakeCamshaftGroup.add(inLobe);

        // 2. Intake Rocker Arm Assembly
        const rGroup = new THREE.Group();
        rGroup.position.set(-0.09, 1.32, vz);
        dohcValveTrainGroup.add(rGroup);

        const pivotBoss = createZCylinder(0.022, 0.022, 0.026, 16, darkBlockMat);
        rGroup.add(pivotBoss);

        const followerArm = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.014, 0.018), machinedSteelMat);
        followerArm.position.set(-0.03, 0.022, 0);
        followerArm.rotation.z = -Math.PI / 6;
        rGroup.add(followerArm);

        const roller = createZCylinder(0.012, 0.012, 0.02, 16, chromeMat);
        roller.position.set(-0.055, 0.038, 0);
        rGroup.add(roller);

        const valveArm = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.012, 0.016), machinedSteelMat);
        valveArm.position.set(0.018, -0.015, 0);
        valveArm.rotation.z = Math.PI / 8;
        rGroup.add(valveArm);

        const tappetScrew = createZCylinder(0.006, 0.006, 0.024, 6, chromeMat);
        tappetScrew.position.set(0.034, -0.025, 0);
        rGroup.add(tappetScrew);

        rockerArms.push({ mesh: rGroup, cylIndex: i, type: "INTAKE" });
    }

    // Exhaust Cam Lobes & Rocker Arms (2 per cylinder)
    for (let zOff of [0.05, -0.05]) {
        const vz = zPos + zOff;

        // 1. Exhaust Cam Lobe on Exhaust Camshaft
        const exLobe = createCamLobeMesh(machinedSteelMat);
        exLobe.position.set(0, 0, vz);
        exLobe.rotation.z = camLobePhase + (3 * Math.PI) / 4;
        exhaustCamshaftGroup.add(exLobe);

        // 2. Exhaust Rocker Arm Assembly
        const rGroup = new THREE.Group();
        rGroup.position.set(0.09, 1.32, vz);
        dohcValveTrainGroup.add(rGroup);

        const pivotBoss = createZCylinder(0.022, 0.022, 0.026, 16, darkBlockMat);
        rGroup.add(pivotBoss);

        const followerArm = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.014, 0.018), machinedSteelMat);
        followerArm.position.set(0.03, 0.022, 0);
        followerArm.rotation.z = Math.PI / 6;
        rGroup.add(followerArm);

        const roller = createZCylinder(0.012, 0.012, 0.02, 16, chromeMat);
        roller.position.set(0.055, 0.038, 0);
        rGroup.add(roller);

        const valveArm = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.012, 0.016), machinedSteelMat);
        valveArm.position.set(-0.018, -0.015, 0);
        valveArm.rotation.z = -Math.PI / 8;
        rGroup.add(valveArm);

        const tappetScrew = createZCylinder(0.006, 0.006, 0.024, 6, chromeMat);
        tappetScrew.position.set(-0.034, -0.025, 0);
        rGroup.add(tappetScrew);

        rockerArms.push({ mesh: rGroup, cylIndex: i, type: "EXHAUST" });
    }
}

// ---------------------------------------------------------------------------
// D. Iconic Organic Crimson Red DOHC Valve Cover & Ignition Coils
// ---------------------------------------------------------------------------
const valveCoverGroup = new THREE.Group();
valveCoverGroup.name = "VALVE_COVER";
engineMasterGroup.add(valveCoverGroup);

// Cylinder Head Plate Base
const headPlate = new THREE.Mesh(
    new THREE.BoxGeometry(0.64, 0.1, 1.4),
    castAluminumMat
);
headPlate.position.set(0, 1.18, 0);
headPlate.castShadow = true;
valveCoverGroup.add(headPlate);

// Organic Beveled Crimson Red DOHC Valve Cover
const organicCoverMesh = createOrganicValveCoverMesh(0.58, 0.22, 1.36, valveCoverRedMat);
organicCoverMesh.position.set(0, 1.29, 0);
organicCoverMesh.castShadow = true;
valveCoverGroup.add(organicCoverMesh);

// Twin Longitudinal DOHC Cam Humps
for (let humpX of [-0.15, 0.15]) {
    const hump = createZCylinder(0.12, 0.12, 1.36, 24, valveCoverRedMat);
    hump.position.set(humpX, 1.40, 0);
    hump.castShadow = true;
    valveCoverGroup.add(hump);
}

// Chrome Oil Filler Cap & Neck
const oilNeck = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.04, 20), castAluminumMat);
oilNeck.position.set(0.18, 1.44, 0.45);
valveCoverGroup.add(oilNeck);

const oilCap = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.04, 20), chromeMat);
oilCap.position.set(0.18, 1.48, 0.45);
valveCoverGroup.add(oilCap);

// 4 Active Ignition Coil Packs & Red High-Voltage Wires
for (let i = 0; i < 4; i++) {
    const zPos = startZ - i * cylinderSpacing;

    const coil = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.06, 0.08), darkBlockMat);
    coil.position.set(0, 1.44, zPos);
    valveCoverGroup.add(coil);

    const wire = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.28, 8), redWireMat);
    wire.rotation.z = Math.PI / 3;
    wire.position.set(-0.12, 1.43, zPos);
    valveCoverGroup.add(wire);

    const hexNut = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.04, 6), brassMat);
    hexNut.position.set(0, 1.48, zPos);
    valveCoverGroup.add(hexNut);
}

// Chrome Perimeter Mounting Bolts
for (let bz of [-0.62, -0.31, 0.0, 0.31, 0.62]) {
    for (let bx of [-0.28, 0.28]) {
        const bolt = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.03, 6), chromeMat);
        bolt.position.set(bx, 1.40, bz);
        valveCoverGroup.add(bolt);
    }
}

// ---------------------------------------------------------------------------
// E. Open Front DOHC Timing Chain & Guide Rails
// ---------------------------------------------------------------------------
// E. Open Front DOHC Timing Chain & Guide Rails
// ---------------------------------------------------------------------------
const timingGroup = new THREE.Group();
timingGroup.name = "TIMING_CHAIN";
timingGroup.position.set(0, 0, 0.69);
engineMasterGroup.add(timingGroup);

// Front Aluminum Timing Cover Plate Backing
const timingCoverPlate = new THREE.Mesh(new THREE.BoxGeometry(0.56, 1.15, 0.02), castAluminumMat);
timingCoverPlate.position.set(0, 1.0, -0.01);
timingGroup.add(timingCoverPlate);

// Dual Extruded 36-Teeth Camshaft Gears at Top Front
const leftCamGear = createExtrudedCamGearMesh(0.16, 0.04, 36, machinedSteelMat);
leftCamGear.position.set(-0.16, 1.40, 0);
timingGroup.add(leftCamGear);

const leftCenterNut = createZCylinder(0.045, 0.045, 0.06, 6, chromeMat);
leftCenterNut.position.set(0, 0, 0.03);
leftCamGear.add(leftCenterNut);

const rightCamGear = createExtrudedCamGearMesh(0.16, 0.04, 36, machinedSteelMat);
rightCamGear.position.set(0.16, 1.40, 0);
timingGroup.add(rightCamGear);

const rightCenterNut = createZCylinder(0.045, 0.045, 0.06, 6, chromeMat);
rightCenterNut.position.set(0, 0, 0.03);
rightCamGear.add(rightCenterNut);

// Lower Crankshaft Drive Timing Sprocket
const crankTimingGear = createExtrudedCamGearMesh(0.10, 0.04, 20, machinedSteelMat);
crankTimingGear.position.set(0, 0.52, 0);
timingGroup.add(crankTimingGear);

const crankCenterNut = createZCylinder(0.035, 0.035, 0.06, 6, chromeMat);
crankCenterNut.position.set(0, 0, 0.03);
crankTimingGear.add(crankCenterNut);

// Iconic Blue Timing Chain Guide Rail (Ref CAD Diagram Left Side Match!)
const blueGuideRail = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.82, 0.03), chainGuideBlueMat);
blueGuideRail.rotation.z = Math.PI / 12;
blueGuideRail.position.set(-0.25, 0.96, 0.01);
timingGroup.add(blueGuideRail);

// Right Side Steel Tensioner Guide Rail
const steelTensionerRail = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.82, 0.03), castAluminumMat);
steelTensionerRail.rotation.z = -Math.PI / 12;
steelTensionerRail.position.set(0.25, 0.96, 0.01);
timingGroup.add(steelTensionerRail);

// Closed 3D Timing Chain Loop Curve Path & Roller Link Assembly
function createTimingChainLinkMesh() {
    const linkGroup = new THREE.Group();

    // Outer steel side plates (front and back)
    const plateShape = new THREE.Shape();
    const r = 0.009;
    const w = 0.026;
    plateShape.absarc(-w / 2, 0, r, Math.PI / 2, Math.PI * 1.5, false);
    plateShape.absarc(w / 2, 0, r, -Math.PI / 2, Math.PI / 2, false);

    const extrudeSettings = { depth: 0.004, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 0.001, bevelThickness: 0.001 };
    const plateGeo = new THREE.ExtrudeGeometry(plateShape, extrudeSettings);

    // Front plate
    const frontPlate = new THREE.Mesh(plateGeo, machinedSteelMat);
    frontPlate.position.z = 0.008;
    linkGroup.add(frontPlate);

    // Back plate
    const backPlate = new THREE.Mesh(plateGeo, machinedSteelMat);
    backPlate.position.z = -0.012;
    linkGroup.add(backPlate);

    // 2 Roller Pins connecting plates
    const pinGeo = new THREE.CylinderGeometry(0.005, 0.005, 0.024, 10);
    pinGeo.rotateX(Math.PI / 2);

    const pin1 = new THREE.Mesh(pinGeo, chromeMat);
    pin1.position.set(-w / 2, 0, 0);
    linkGroup.add(pin1);

    const pin2 = new THREE.Mesh(pinGeo, chromeMat);
    pin2.position.set(w / 2, 0, 0);
    linkGroup.add(pin2);

    // Center roller cylinder
    const rollerGeo = new THREE.CylinderGeometry(0.008, 0.008, 0.016, 12);
    rollerGeo.rotateX(Math.PI / 2);
    const roller = new THREE.Mesh(rollerGeo, darkBlockMat);
    linkGroup.add(roller);

    return linkGroup;
}

const timingChainPoints = [
    new THREE.Vector3(0.0, 0.42, 0.02),      // Bottom crank gear bottom arc
    new THREE.Vector3(-0.095, 0.50, 0.02),   // Crank gear bottom-left tangent
    new THREE.Vector3(-0.21, 0.96, 0.02),    // Left guide rail center
    new THREE.Vector3(-0.315, 1.40, 0.02),   // Left cam gear far-left tangent
    new THREE.Vector3(-0.16, 1.555, 0.02),   // Left cam gear top tangent
    new THREE.Vector3(0.0, 1.555, 0.02),     // Top span center
    new THREE.Vector3(0.16, 1.555, 0.02),    // Right cam gear top tangent
    new THREE.Vector3(0.315, 1.40, 0.02),    // Right cam gear far-right tangent
    new THREE.Vector3(0.21, 0.96, 0.02),     // Right tensioner rail center
    new THREE.Vector3(0.095, 0.50, 0.02)     // Crank gear bottom-right tangent
];

const timingChainCurve = new THREE.CatmullRomCurve3(timingChainPoints, true, 'centripetal', 0.25);

const timingChainGroup = new THREE.Group();
timingGroup.add(timingChainGroup);

const numTimingChainLinks = 68;
const timingChainLinks = [];

for (let i = 0; i < numTimingChainLinks; i++) {
    const link = createTimingChainLinkMesh();
    timingChainGroup.add(link);
    timingChainLinks.push(link);
}


// ---------------------------------------------------------------------------
// F. TURBOCHARGER WITH COMPLETE VOLUTE RINGS & TANGENTIAL SNOUTS (Facing Forward +Z)
// ---------------------------------------------------------------------------
const exhaustGroup = new THREE.Group();
exhaustGroup.name = "EXHAUST_HEADERS";
engineMasterGroup.add(exhaustGroup);

// 4 Sweeping 3D Curved Tubular Header Pipes
const headerPipes = [];
for (let i = 0; i < 4; i++) {
    const zPos = startZ - i * cylinderSpacing;

    const curvePoints = [
        new THREE.Vector3(0.31, 1.05, zPos),
        new THREE.Vector3(0.48, 1.02, zPos + (1.5 - i) * 0.04),
        new THREE.Vector3(0.55, 0.78, -0.15),
        new THREE.Vector3(0.52, 0.58, -0.15)
    ];
    const path = new THREE.CatmullRomCurve3(curvePoints);
    const pipeGeo = new THREE.TubeGeometry(path, 24, 0.042, 16, false);
    const pipeMesh = new THREE.Mesh(pipeGeo, exhaustHeaderMat);
    pipeMesh.castShadow = true;
    exhaustGroup.add(pipeMesh);

    // Flange Plate at Head Interface
    const flange = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.12, 0.12), darkBlockMat);
    flange.position.set(0.32, 1.05, zPos);
    exhaustGroup.add(flange);

    headerPipes.push(pipeMesh);
}

// 4-into-1 Exhaust Header Collector Flange Plate
const collectorFlange = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.04, 0.18), darkBlockMat);
collectorFlange.position.set(0.52, 0.56, -0.15);
exhaustGroup.add(collectorFlange);

// TURBOCHARGER ASSEMBLY (COMPLETE VOLUTE RINGS WITH TANGENTIAL SNOUTS!)
const turboGroup = new THREE.Group();
turboGroup.position.set(0.52, 0.44, -0.15);
exhaustGroup.add(turboGroup);

// 1. Cold Side Compressor Scroll Housing (Polished Chrome COMPLETE 360° VOLUTE RING)
const compressorScrollGeo = new THREE.TorusGeometry(0.15, 0.068, 24, 48, Math.PI * 2.0); // Full 360° Volute Ring
const compressorScroll = new THREE.Mesh(compressorScrollGeo, chromeMat);
compressorScroll.position.set(0, 0, 0.08);
compressorScroll.castShadow = true;
turboGroup.add(compressorScroll);

// Tangential Discharge Snout Tube exiting top of compressor ring
const compressorSnout = new THREE.Mesh(
    new THREE.CylinderGeometry(0.048, 0.055, 0.16, 24),
    chromeMat
);
compressorSnout.rotation.z = -Math.PI / 4; // Rotated tangentially off top right of ring
compressorSnout.position.set(0.10, 0.12, 0.08);
turboGroup.add(compressorSnout);

// Circular Air Intake Mouthpiece Trumpet Bezel
const compressorInlet = createZCylinder(0.088, 0.088, 0.08, 24, chromeMat);
compressorInlet.position.set(0, 0, 0.16);
turboGroup.add(compressorInlet);

// Compressor Impeller Wheel Blades inside forward-facing inlet
const impellerGroup = new THREE.Group();
impellerGroup.position.set(0, 0, 0.16);
turboGroup.add(impellerGroup);

for (let b = 0; b < 8; b++) {
    const bladeHolder = new THREE.Group();
    bladeHolder.rotation.z = (b * Math.PI) / 4;
    impellerGroup.add(bladeHolder);

    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.008, 0.04), chromeMat);
    blade.position.set(0.032, 0, 0);
    bladeHolder.add(blade);
}

// Backplate Fastener Bolts (8 Perimeter Bolts)
for (let ab = 0; ab < 8; ab++) {
    const angle = (ab * Math.PI) / 4;
    const bolt = createZCylinder(0.01, 0.01, 0.02, 6, chromeMat);
    bolt.position.set(0.15 * Math.cos(angle), 0.15 * Math.sin(angle), 0.02);
    turboGroup.add(bolt);
}

// 2. CHRA (Center Housing Rotating Assembly) Oil Cooled Core Cartridge
const chraHousing = createZCylinder(0.075, 0.075, 0.10, 24, brassMat);
chraHousing.position.set(0, 0, 0);
turboGroup.add(chraHousing);

// Stainless Steel V-Band Clamp Rings
const vBandFront = new THREE.Mesh(new THREE.TorusGeometry(0.082, 0.012, 16, 32), chromeMat);
vBandFront.position.set(0, 0, 0.045);
turboGroup.add(vBandFront);

const vBandRear = new THREE.Mesh(new THREE.TorusGeometry(0.080, 0.012, 16, 32), chromeMat);
vBandRear.position.set(0, 0, -0.045);
turboGroup.add(vBandRear);

// Braided Stainless Steel Oil Supply Line & AN Fitting
const oilLineInlet = createZCylinder(0.015, 0.015, 0.04, 6, chromeMat);
oilLineInlet.position.set(0, 0.075, 0);
turboGroup.add(oilLineInlet);

const oilFeedHose = new THREE.Mesh(
    new THREE.CylinderGeometry(0.010, 0.010, 0.28, 12),
    machinedSteelMat
);
oilFeedHose.position.set(-0.06, 0.20, 0);
oilFeedHose.rotation.z = Math.PI / 4;
turboGroup.add(oilFeedHose);

// 3. Hot Side Turbine Scroll Housing (Dark Cast Iron COMPLETE 360° VOLUTE RING)
const turbineScrollGeo = new THREE.TorusGeometry(0.14, 0.060, 24, 48, Math.PI * 2.0); // Full 360° Volute Ring
const turbineScroll = new THREE.Mesh(turbineScrollGeo, darkBlockMat);
turbineScroll.position.set(0, 0, -0.08);
turbineScroll.castShadow = true;
turboGroup.add(turbineScroll);

// Tangential Exhaust Entry Snout Duct connecting Collector Flange to Turbine Scroll Ring
const turbineEntrySnout = new THREE.Mesh(
    new THREE.BoxGeometry(0.10, 0.12, 0.08),
    darkBlockMat
);
turbineEntrySnout.position.set(0.0, 0.10, -0.08);
turboGroup.add(turbineEntrySnout);

// Rear Exhaust Downpipe Cone
const exhaustDownpipeCone = createZCylinder(0.075, 0.065, 0.32, 20, exhaustHeaderMat);
exhaustDownpipeCone.position.set(0, 0, -0.24);
turboGroup.add(exhaustDownpipeCone);

// Exhaust Downpipe Mounting Flange Plate
const downpipeFlange = createZCylinder(0.09, 0.09, 0.02, 6, darkBlockMat);
downpipeFlange.position.set(0, 0, -0.16);
turboGroup.add(downpipeFlange);

// 4. Wastegate Actuator Canister & Stainless Control Rod
const wastegateCanister = new THREE.Mesh(
    new THREE.CylinderGeometry(0.038, 0.038, 0.08, 16),
    brassMat
);
wastegateCanister.rotation.x = Math.PI / 2;
wastegateCanister.position.set(-0.16, 0.10, 0.02);
turboGroup.add(wastegateCanister);

const wastegateRod = new THREE.Mesh(
    new THREE.CylinderGeometry(0.006, 0.006, 0.16, 8),
    chromeMat
);
wastegateRod.rotation.x = Math.PI / 2;
wastegateRod.position.set(-0.16, 0.10, -0.06);
turboGroup.add(wastegateRod);

// Wastegate Lever Arm on Turbine Housing
const wastegateArm = new THREE.Mesh(
    new THREE.BoxGeometry(0.012, 0.04, 0.012),
    darkBlockMat
);
wastegateArm.position.set(-0.16, 0.10, -0.13);
turboGroup.add(wastegateArm);

// 5. Intercooler Charge Air Pipe running from Turbo Compressor Snout around block into Intake
const chargePipePoints = [
    new THREE.Vector3(0.62, 0.56, -0.07),    // Top of Turbo Compressor Snout
    new THREE.Vector3(0.55, 0.25, 0.08),     // Down underneath right side
    new THREE.Vector3(0.0, 0.22, 0.08),     // Under engine block center
    new THREE.Vector3(-0.45, 0.25, 0.08),    // Under left side
    new THREE.Vector3(-0.48, 0.65, 0.35),    // Up left side
    new THREE.Vector3(-0.48, 1.08, 0.65)     // Into Throttle Body entry
];
const chargePipePath = new THREE.CatmullRomCurve3(chargePipePoints);
const chargePipeGeo = new THREE.TubeGeometry(chargePipePath, 48, 0.042, 16, false);
const chargePipeMesh = new THREE.Mesh(chargePipeGeo, chargePipeMat);
chargePipeMesh.castShadow = true;
engineMasterGroup.add(chargePipeMesh);

// Black Rubber Hose Couplers
for (let pt of [chargePipePoints[0], chargePipePoints[5]]) {
    const coupler = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.048, 0.07, 16), serpentineBeltMat);
    coupler.position.copy(pt);
    engineMasterGroup.add(coupler);
}

// ---------------------------------------------------------------------------
// G-PRE. AIR INTAKE SYSTEM — FRONT-BOTTOM INTERCOOLER + CLEAN BENT CHARGE PIPES
// Path: Turbo compressor outlet → Hot pipe (bent) → Intercooler (front face, bottom)
//       → Cold pipe (bent) → Throttle body
// Turbo outlet ≈ world (0.62, 0.56, -0.07)  |  Throttle body ≈ world (-0.48, 1.08, 0.65)
// ---------------------------------------------------------------------------
const airIntakeSystemGroup = new THREE.Group();
airIntakeSystemGroup.name = "AIR_INTAKE_INTERCOOLER";
engineMasterGroup.add(airIntakeSystemGroup);

// ── Materials ──────────────────────────────────────────────────────────────
// (Defined in main PBR Materials section at top)

// Helper: create a straight pipe segment between two points
function makePipeSegment(x1, y1, z1, x2, y2, z2, radius, mat) {
    const pts = [new THREE.Vector3(x1, y1, z1), new THREE.Vector3(x2, y2, z2)];
    const curve = new THREE.CatmullRomCurve3(pts);
    const geo = new THREE.TubeGeometry(curve, 4, radius, 16, false);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    return mesh;
}

// Helper: blue silicone coupler ring at a point
function makeCoupler(x, y, z, rx, ry, rz, radius, mat) {
    const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(radius + 0.008, radius + 0.008, 0.065, 20),
        mat
    );
    mesh.rotation.set(rx, ry, rz);
    mesh.position.set(x, y, z);
    return mesh;
}

const PR = 0.044; // pipe radius

// ═══════════════════════════════════════════════════════════════════════════
// 1. FRONT-MOUNTED INTERCOOLER — sits at the front face, bottom of engine
//    (z = 0.90, y = 0.28, centred on X-axis, width 1.20)
// ═══════════════════════════════════════════════════════════════════════════
const IC_X = 0.0;   // centred
const IC_Y = 0.28;  // bottom of engine height
const IC_Z = 0.92;  // front face of engine

const intercoolerGroup = new THREE.Group();
intercoolerGroup.position.set(IC_X, IC_Y, IC_Z);
airIntakeSystemGroup.add(intercoolerGroup);

// Core body (faces forward, depth along Z)
const icCoreBody = new THREE.Mesh(
    new THREE.BoxGeometry(1.20, 0.26, 0.13),
    intercoolerBodyMat
);
icCoreBody.castShadow = true;
intercoolerGroup.add(icCoreBody);

// Left & right end tanks
const icTankGeo = new THREE.BoxGeometry(0.13, 0.28, 0.17);
const icTankL = new THREE.Mesh(icTankGeo, castAluminumMat);
icTankL.position.set(-0.665, 0, 0);
intercoolerGroup.add(icTankL);

const icTankR = new THREE.Mesh(icTankGeo, castAluminumMat);
icTankR.position.set(0.665, 0, 0);
intercoolerGroup.add(icTankR);

// Horizontal cooling fins
for (let f = -4; f <= 4; f++) {
    const fin = new THREE.Mesh(
        new THREE.BoxGeometry(1.18, 0.004, 0.135),
        intercoolerFinMat
    );
    fin.position.set(0, f * 0.026, 0);
    intercoolerGroup.add(fin);
}

// Vertical internal flow tubes
for (let v = -5; v <= 5; v++) {
    const tube = new THREE.Mesh(
        new THREE.BoxGeometry(0.007, 0.252, 0.125),
        intercoolerBodyMat
    );
    tube.position.set(v * 0.10, 0, 0);
    intercoolerGroup.add(tube);
}

// Top & bottom mounting brackets
for (const s of [-1, 1]) {
    const brk = new THREE.Mesh(
        new THREE.BoxGeometry(1.30, 0.028, 0.055),
        chromeMat
    );
    brk.position.set(0, s * 0.148, -0.04);
    intercoolerGroup.add(brk);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. HOT-SIDE CHARGE PIPE  Turbo outlet → Intercooler right tank
//    Turbo compressor snout outlet ≈ (0.65, 0.56, -0.07)
//    Route: forward along Z, then drop down to IC height, enter IC right tank
//    Bend points:
//      A (0.65, 0.56, -0.07)  → turbo outlet
//      B (0.65, 0.56,  0.92)  → forward (same X, same Y, at IC front z)
//      C (0.65, 0.28,  0.92)  → drop down to IC height  → enters IC right tank
// ═══════════════════════════════════════════════════════════════════════════

// Segment A→B: horizontal forward run
const hotPipeAB = makePipeSegment(0.65, 0.56, -0.07, 0.65, 0.56, 0.85, PR, chargePipeMat);
airIntakeSystemGroup.add(hotPipeAB);

// Bend coupler at B (at the corner going downward)
const hotBendB = makeCoupler(0.65, 0.56, 0.86, 0, 0, Math.PI / 2, PR, siliconeCouplerMat);
airIntakeSystemGroup.add(hotBendB);

// Segment B→C: vertical drop into IC right tank
const hotPipeBC = makePipeSegment(0.65, 0.56, 0.92, 0.665, 0.28, 0.92, PR, chargePipeMat);
airIntakeSystemGroup.add(hotPipeBC);

// Coupler at turbo outlet
const hotCouplerTurbo = makeCoupler(0.65, 0.56, -0.06, Math.PI / 2, 0, 0, PR, siliconeCouplerMat);
airIntakeSystemGroup.add(hotCouplerTurbo);

// Coupler at IC right tank inlet (on top of right tank)
const hotCouplerIC = makeCoupler(0.665, 0.42, 0.92, 0, 0, 0, PR, siliconeCouplerMat);
airIntakeSystemGroup.add(hotCouplerIC);

// ═══════════════════════════════════════════════════════════════════════════
// 3. COLD-SIDE CHARGE PIPE  Intercooler left tank → Throttle body
//    IC left tank top ≈ (-0.665, 0.42, 0.92)
//    Throttle body inlet ≈ (-0.48, 1.08, 0.65)
//    Route: rise up from IC left tank, then run back (–Z) to throttle body
//    Bend points:
//      D (-0.665, 0.42, 0.92)  → IC left tank top
//      E (-0.665, 1.08, 0.92)  → rise up to intake manifold height
//      F (-0.48,  1.08, 0.66)  → run back to throttle body
// ═══════════════════════════════════════════════════════════════════════════

// Coupler at IC left tank outlet (on top of left tank)
const coldCouplerIC = makeCoupler(-0.665, 0.42, 0.92, 0, 0, 0, PR, siliconeCouplerMat);
airIntakeSystemGroup.add(coldCouplerIC);

// Segment D→E: rise straight up
const coldPipeDE = makePipeSegment(-0.665, 0.42, 0.92, -0.665, 1.08, 0.92, PR, chargePipeMat);
airIntakeSystemGroup.add(coldPipeDE);

// Bend coupler at E (corner going backward toward throttle)
const coldBendE = makeCoupler(-0.665, 1.08, 0.90, 0, 0, Math.PI / 2, PR, siliconeCouplerMat);
airIntakeSystemGroup.add(coldBendE);

// Segment E→F: run back toward throttle body
const coldPipeEF = makePipeSegment(-0.665, 1.08, 0.92, -0.52, 1.08, 0.66, PR, chargePipeMat);
airIntakeSystemGroup.add(coldPipeEF);

// Coupler at throttle body inlet
const coldCouplerThrottle = makeCoupler(-0.52, 1.08, 0.68, Math.PI / 2, 0, 0, PR, siliconeCouplerMat);
airIntakeSystemGroup.add(coldCouplerThrottle);

// ---------------------------------------------------------------------------
// G. INTAKE MANIFOLD WITH OPEN FRONT INLET VELOCITY STACK
// ---------------------------------------------------------------------------
const intakeGroup = new THREE.Group();
intakeGroup.name = "INTAKE_THROTTLE";
engineMasterGroup.add(intakeGroup);

// Main Intake Plenum Log Tube (Aluminum -X side)
const intakePlenum = createZCylinder(0.11, 0.11, 1.32, 24, castAluminumMat);
intakePlenum.position.set(-0.48, 1.08, 0);
intakePlenum.castShadow = true;
intakeGroup.add(intakePlenum);

// Rear End Cap for Plenum Log
const plenumRearCap = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 16), castAluminumMat);
plenumRearCap.position.set(-0.48, 1.08, -0.66);
intakeGroup.add(plenumRearCap);

// OPEN FRONT INLET APERTURE / VELOCITY STACK
const openInletBezel = new THREE.Mesh(
    new THREE.TorusGeometry(0.11, 0.018, 16, 32),
    chromeMat
);
openInletBezel.position.set(-0.48, 1.08, 0.66);
intakeGroup.add(openInletBezel);

// Open Inner Cavity (Dark recessed interior)
const openInletCavity = createZCylinder(0.092, 0.092, 0.06, 32, darkBlockMat);
openInletCavity.position.set(-0.48, 1.08, 0.64);
intakeGroup.add(openInletCavity);

// 4 Curved Aluminum Intake Runner Pipes
for (let i = 0; i < 4; i++) {
    const zPos = startZ - i * cylinderSpacing;
    const runnerPoints = [
        new THREE.Vector3(-0.48, 1.08, zPos),
        new THREE.Vector3(-0.42, 1.08, zPos),
        new THREE.Vector3(-0.31, 1.05, zPos)
    ];
    const runnerPath = new THREE.CatmullRomCurve3(runnerPoints);
    const runnerGeo = new THREE.TubeGeometry(runnerPath, 16, 0.042, 16, false);
    const runnerMesh = new THREE.Mesh(runnerGeo, castAluminumMat);
    runnerMesh.castShadow = true;
    intakeGroup.add(runnerMesh);

    // Flange at head interface
    const flange = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.11, 0.11), darkBlockMat);
    flange.position.set(-0.32, 1.05, zPos);
    intakeGroup.add(flange);
}

// Throttle Body Assembly
const throttleBody = createZCylinder(0.085, 0.085, 0.18, 20, castAluminumMat);
throttleBody.position.set(-0.48, 1.08, 0.65);
intakeGroup.add(throttleBody);

// Throttle Linkage Lever Arm
const throttleLever = new THREE.Mesh(
    new THREE.BoxGeometry(0.016, 0.14, 0.025),
    chromeMat
);
throttleLever.position.set(-0.57, 1.08, 0.65);
intakeGroup.add(throttleLever);

// ---------------------------------------------------------------------------
// H. REALISTIC ALTERNATOR WITH VENTILATION LINES & PULLEY FAN
// ---------------------------------------------------------------------------
const frontDriveGroup = new THREE.Group();
frontDriveGroup.name = "ALTERNATOR_SERPENTINE";
frontDriveGroup.position.set(0, 0, 0.77);
engineMasterGroup.add(frontDriveGroup);

// 1. Crankshaft Double V-Pulley
const crankPulley = createVPulleyMesh(0.22, 0.08, darkPulleyMat);
crankPulley.position.set(0, 0.52, 0);
crankPulley.castShadow = true;
frontDriveGroup.add(crankPulley);

// 2. Alternator Assembly
const alternatorGroup = new THREE.Group();
alternatorGroup.position.set(-0.38, 0.75, -0.11);
frontDriveGroup.add(alternatorGroup);

const alternatorHousing = createZCylinder(0.17, 0.17, 0.22, 32, castAluminumMat);
alternatorGroup.add(alternatorHousing);

// 12 Longitudinal Dark Cooling Ventilation Slot Lines
for (let s = 0; s < 12; s++) {
    const slotAngle = (s * 2 * Math.PI) / 12;
    const slotLine = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.012, 0.18), darkBlockMat);
    slotLine.position.set(Math.cos(slotAngle) * 0.172, Math.sin(slotAngle) * 0.172, 0);
    alternatorGroup.add(slotLine);
}

// Internal Copper Coil Rotor
const copperCoil = createZCylinder(0.14, 0.14, 0.16, 24, brassMat);
copperCoil.position.z = -0.02;
alternatorGroup.add(copperCoil);

// Small Cooling Impeller Fan attached behind pulley
const altFanMesh = new THREE.Group();
altFanMesh.position.set(0, 0, 0.07);
alternatorGroup.add(altFanMesh);

const altFanDisc = createZCylinder(0.15, 0.15, 0.012, 16, darkBlockMat);
altFanMesh.add(altFanDisc);

for (let b = 0; b < 8; b++) {
    const bladeHolder = new THREE.Group();
    bladeHolder.rotation.z = (b * 2 * Math.PI) / 8;
    altFanMesh.add(bladeHolder);

    const fanBlade = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.012, 0.025), castAluminumMat);
    fanBlade.rotation.x = Math.PI / 4;
    fanBlade.position.set(0.12, 0, 0);
    bladeHolder.add(fanBlade);
}

// Alternator Belt Pulley
const altPulley = createVPulleyMesh(0.11, 0.08, darkPulleyMat);
altPulley.position.set(0, 0, 0.11);
alternatorGroup.add(altPulley);

// 3. Water Pump Pulley
const wpPulley = createVPulleyMesh(0.15, 0.08, darkPulleyMat);
wpPulley.position.set(0.35, 0.85, 0);
frontDriveGroup.add(wpPulley);

// 4. Tensioner Pulley
const tensionerPulley = createVPulleyMesh(0.09, 0.07, darkPulleyMat);
tensionerPulley.position.set(0.22, 0.42, 0);
frontDriveGroup.add(tensionerPulley);

// PERFECT 3D CLOSED SERPENTINE BELT PATH
const beltPathPoints = [
    new THREE.Vector3(-0.49, 0.75, 0.0),
    new THREE.Vector3(-0.38, 0.86, 0.0),
    new THREE.Vector3(0.35, 0.98, 0.0),
    new THREE.Vector3(0.50, 0.85, 0.0),
    new THREE.Vector3(0.31, 0.42, 0.0),
    new THREE.Vector3(0.0, 0.30, 0.0),
    new THREE.Vector3(-0.22, 0.52, 0.0),
];
const beltClosedCurve = new THREE.CatmullRomCurve3(beltPathPoints, true, "centripetal");
const beltPathGeo = new THREE.TubeGeometry(beltClosedCurve, 64, 0.016, 12, true);
const beltMesh = new THREE.Mesh(beltPathGeo, serpentineBeltMat);
frontDriveGroup.add(beltMesh);

// ---------------------------------------------------------------------------
// I. SINGLE MAIN AERO PROPELLER (Rear — Mounted on Power Reduction Output Flange)
// ---------------------------------------------------------------------------
// propGroup is declared and added AFTER outputFlangeGroup is built (see Section J below)
// It is attached to outputFlangeGroup so it rotates with the reduction drive flange


// ---------------------------------------------------------------------------
// J. VERTICAL Y-AXIS AERO REDUCTION GEARBOX & TOP ROTATING OUTPUT FLANGE
// ---------------------------------------------------------------------------
const rearEngineGroup = new THREE.Group();
rearEngineGroup.name = "REAR_FLYWHEEL";
rearEngineGroup.position.set(0, 0.52, -0.72);
engineMasterGroup.add(rearEngineGroup);

// 1. Lower Flywheel Disc & Ring Gear (Internal - Attached to Engine Crankshaft at y = 0.52)
const flywheelDiscGroup = new THREE.Group();
flywheelDiscGroup.position.z = -0.04;
rearEngineGroup.add(flywheelDiscGroup);

const flywheelDisc = createZCylinder(0.33, 0.33, 0.05, 32, machinedSteelMat);
flywheelDisc.castShadow = true;
flywheelDiscGroup.add(flywheelDisc);

const ringGearTorus = new THREE.TorusGeometry(0.335, 0.012, 16, 64);
const ringGearMesh = new THREE.Mesh(ringGearTorus, darkBlockMat);
flywheelDiscGroup.add(ringGearMesh);

// 2. Heavy Engine Backing Mount Flange Plate & Flywheel-to-Gearbox Connection
const engineMountFlange = new THREE.Mesh(
    new THREE.TorusGeometry(0.40, 0.02, 16, 48),
    castAluminumMat
);
engineMountFlange.position.z = -0.02;
rearEngineGroup.add(engineMountFlange);

for (let b = 0; b < 10; b++) {
    const angle = (b * 2 * Math.PI) / 10;
    const stud = createZCylinder(0.015, 0.015, 0.04, 6, chromeMat);
    stud.position.set(Math.cos(angle) * 0.40, Math.sin(angle) * 0.40, -0.02);
    rearEngineGroup.add(stud);
}

// Solid Heavy Bellhousing Tunnel Adapter connecting Flywheel Housing to Reduction Gearbox!
const bellhousingTunnel = createZCylinder(0.28, 0.36, 0.22, 32, castAluminumMat);
bellhousingTunnel.position.set(0, 0, -0.15);
bellhousingTunnel.castShadow = true;
rearEngineGroup.add(bellhousingTunnel);

// Front Flange Ring attaching Tunnel to Flywheel Housing
const tunnelFrontFlange = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.016, 16, 48), chromeMat);
tunnelFrontFlange.position.set(0, 0, -0.05);
rearEngineGroup.add(tunnelFrontFlange);

// Rear Flange Ring attaching Tunnel to Reduction Gearbox Lower Chamber
const tunnelRearFlange = new THREE.Mesh(new THREE.TorusGeometry(0.285, 0.016, 16, 48), chromeMat);
tunnelRearFlange.position.set(0, 0, -0.25);
rearEngineGroup.add(tunnelRearFlange);

for (let b = 0; b < 8; b++) {
    const angle = (b * 2 * Math.PI) / 8;
    const boltF = createZCylinder(0.012, 0.012, 0.035, 6, chromeMat);
    boltF.position.set(Math.cos(angle) * 0.36, Math.sin(angle) * 0.36, -0.05);
    rearEngineGroup.add(boltF);

    const boltR = createZCylinder(0.012, 0.012, 0.035, 6, chromeMat);
    boltR.position.set(Math.cos(angle) * 0.285, Math.sin(angle) * 0.285, -0.25);
    rearEngineGroup.add(boltR);
}

// Rotating Input Drive Shaft Core extending from Flywheel into Lower Gearbox Chamber!
const flywheelInputShaft = createZCylinder(0.12, 0.12, 0.20, 24, machinedSteelMat);
flywheelInputShaft.position.set(0, 0, -0.14);
flywheelDiscGroup.add(flywheelInputShaft);

// Torsional Vibration Damper Flex-Plate Disc (Rotating with Flywheel)
const flexPlateDamper = createZCylinder(0.26, 0.26, 0.03, 32, darkBlockMat);
flexPlateDamper.position.set(0, 0, -0.10);
flywheelDiscGroup.add(flexPlateDamper);

for (let b = 0; b < 6; b++) {
    const angle = (b * 2 * Math.PI) / 6;
    const bolt = createZCylinder(0.014, 0.014, 0.035, 6, chromeMat);
    bolt.position.set(Math.cos(angle) * 0.10, Math.sin(angle) * 0.10, -0.11);
    flywheelDiscGroup.add(bolt);
}

// 3. VERTICAL Y-AXIS DUAL-SHAFT REDUCTION GEARBOX (Crystal-Clear Transparent Casing)
const gearboxBodyGroup = new THREE.Group();
gearboxBodyGroup.name = "TIMING_CHAIN";
gearboxBodyGroup.position.set(0, 0, -0.28);
rearEngineGroup.add(gearboxBodyGroup);

// Transparent Glass Lower Input Chamber (Enclosing Crankshaft Drive Gear at y = 0)
const lowerChamber = createZCylinder(0.22, 0.22, 0.18, 32, gearboxGlassMat);
lowerChamber.position.set(0, 0, 0);
gearboxBodyGroup.add(lowerChamber);

// Transparent Glass Upper Output Chamber (Enclosing Propeller Driven Gear at y = 0.36)
const upperChamber = createZCylinder(0.28, 0.28, 0.18, 32, gearboxGlassMat);
upperChamber.position.set(0, 0.36, 0);
gearboxBodyGroup.add(upperChamber);

// Transparent Glass Vertical Connecting Bridge Housing
const connectingBridge = new THREE.Mesh(
    new THREE.BoxGeometry(0.48, 0.42, 0.17),
    gearboxGlassMat
);
connectingBridge.position.set(0, 0.18, 0);
gearboxBodyGroup.add(connectingBridge);

// Solid Cast Aluminum Split Seam Flange Rail & Perimeter Mounting Bolts
const seamFlangeRail = new THREE.Mesh(
    new THREE.BoxGeometry(0.52, 0.46, 0.02),
    castAluminumMat
);
seamFlangeRail.position.set(0, 0.18, 0.09);
gearboxBodyGroup.add(seamFlangeRail);

const seamFlangeRailBack = new THREE.Mesh(
    new THREE.BoxGeometry(0.52, 0.46, 0.02),
    castAluminumMat
);
seamFlangeRailBack.position.set(0, 0.18, -0.09);
gearboxBodyGroup.add(seamFlangeRailBack);

const boltCoords = [
    [-0.24, 0.38], [0.24, 0.38], [-0.24, -0.06], [0.24, -0.06],
    [0.0, 0.40], [0.0, -0.08], [-0.25, 0.18], [0.25, 0.18]
];
for (let pt of boltCoords) {
    const hexBolt = createZCylinder(0.012, 0.012, 0.20, 6, chromeMat);
    hexBolt.position.set(pt[0], pt[1], 0);
    gearboxBodyGroup.add(hexBolt);
}

// ---------------------------------------------------------------------------
// INTERNAL REDUCTION SPUR GEAR PAIR (Meshed Top & Bottom Gears with Teeth!)
// ---------------------------------------------------------------------------
// 1. Bottom Crankshaft Drive Gear (20 Teeth, connected to Crankshaft Flywheel Input Shaft)
const lowerDriveGear = createExtrudedCamGearMesh(0.15, 0.05, 20, machinedSteelMat);
lowerDriveGear.position.set(0, 0, 0);
lowerDriveGear.castShadow = true;
gearboxBodyGroup.add(lowerDriveGear);

const lowerGearCenterNut = createZCylinder(0.04, 0.04, 0.07, 6, chromeMat);
lowerGearCenterNut.position.set(0, 0, 0.02);
lowerDriveGear.add(lowerGearCenterNut);

// 2. Top Propeller Driven Gear (30 Teeth, connected directly to Propeller Drive Shaft)
const upperDrivenGear = createExtrudedCamGearMesh(0.22, 0.05, 30, machinedSteelMat);
upperDrivenGear.position.set(0, 0.36, 0); // Meshes directly at top with 20-teeth gear!
upperDrivenGear.castShadow = true;
gearboxBodyGroup.add(upperDrivenGear);

const upperGearCenterNut = createZCylinder(0.04, 0.04, 0.07, 6, chromeMat);
upperGearCenterNut.position.set(0, 0, 0.02);
upperDrivenGear.add(upperGearCenterNut);

// 4. ROTATING DARK PROPELLER DRIVE OUTPUT FLANGE & HEAVY CONNECTING DRIVE SHAFT
const outputFlangeGroup = new THREE.Group();
outputFlangeGroup.position.set(0, 0.36, -0.54); // Global y = 0.88, z = -1.26
rearEngineGroup.add(outputFlangeGroup);

// Upper Shaft End Collar
const shaftCollar = createZCylinder(0.13, 0.15, 0.04, 24, darkBlockMat);
shaftCollar.position.set(0, 0, 0.02);
outputFlangeGroup.add(shaftCollar);

// Main Heavy Dark Anodized Forged Steel Output Flange Disc (ROTATING AT TOP!)
const darkFlangeDisc = createZCylinder(0.20, 0.20, 0.04, 32, darkFlangeMat);
darkFlangeDisc.position.set(0, 0, -0.02);
darkFlangeDisc.castShadow = true;
outputFlangeGroup.add(darkFlangeDisc);

// Protruding Central Hub Cup
const centerHubCup = createZCylinder(0.095, 0.095, 0.04, 24, darkFlangeMat);
centerHubCup.position.set(0, 0, -0.04);
outputFlangeGroup.add(centerHubCup);

// Recessed Central Bore Hole
const centralBoreHole = createZCylinder(0.055, 0.055, 0.042, 24, darkBlockMat);
centralBoreHole.position.set(0, 0, -0.04);
outputFlangeGroup.add(centralBoreHole);

// 6 Outer Perimeter Circular Drive Holes
for (let b = 0; b < 6; b++) {
    const angle = (b * 2 * Math.PI) / 6;
    const holeX = Math.cos(angle) * 0.138;
    const holeY = Math.sin(angle) * 0.138;

    const driveHole = createZCylinder(0.024, 0.024, 0.042, 16, darkBlockMat);
    driveHole.position.set(holeX, holeY, -0.02);
    outputFlangeGroup.add(driveHole);

    const innerBoltX = Math.cos(angle + Math.PI / 6) * 0.075;
    const innerBoltY = Math.sin(angle + Math.PI / 6) * 0.075;
    const hexBolt = createZCylinder(0.012, 0.012, 0.048, 6, chromeMat);
    hexBolt.position.set(innerBoltX, innerBoltY, -0.02);
    outputFlangeGroup.add(hexBolt);
}

// ---------------------------------------------------------------------------
// PROPELLER DRIVE SHAFT CONNECTION (Connecting Gearbox to Propeller Hub)
// ---------------------------------------------------------------------------
// 1. Continuous Heavy Forged Steel Propeller Drive Shaft
const mainPropellerDriveShaft = createZCylinder(0.078, 0.078, 0.55, 32, machinedSteelMat);
mainPropellerDriveShaft.position.set(0, 0, -0.16);
mainPropellerDriveShaft.castShadow = true;
outputFlangeGroup.add(mainPropellerDriveShaft);

// 2. Heavy Anodized Dark Steel Splined Drive Coupling Sleeve
const splinedCouplingSleeve = createZCylinder(0.096, 0.096, 0.28, 24, darkFlangeMat);
splinedCouplingSleeve.position.set(0, 0, -0.15);
splinedCouplingSleeve.castShadow = true;
outputFlangeGroup.add(splinedCouplingSleeve);

// 3. Front Gearbox Locking Collar
const frontLockingCollar = createZCylinder(0.125, 0.125, 0.06, 24, chromeMat);
frontLockingCollar.position.set(0, 0, -0.04);
outputFlangeGroup.add(frontLockingCollar);

// 4. Rear Propeller Hub Adaptor Flange
const rearHubAdaptorCollar = createZCylinder(0.145, 0.145, 0.06, 24, chromeMat);
rearHubAdaptorCollar.position.set(0, 0, -0.24);
rearHubAdaptorCollar.castShadow = true;
outputFlangeGroup.add(rearHubAdaptorCollar);

// 5. Perimeter Shaft Coupling Bolts (6 Chrome Bolts on Rear Adaptor Collar)
for (let c = 0; c < 6; c++) {
    const angle = (c * 2 * Math.PI) / 6;
    const boltX = Math.cos(angle) * 0.11;
    const boltY = Math.sin(angle) * 0.11;
    const cBolt = createZCylinder(0.014, 0.014, 0.07, 6, chromeMat);
    cBolt.position.set(boltX, boltY, -0.24);
    outputFlangeGroup.add(cBolt);
}

// ---------------------------------------------------------------------------
// I (continued). PROPELLER ASSEMBLY — Attached to Rear Output Shaft Connection
// ---------------------------------------------------------------------------
const propGroup = new THREE.Group();
propGroup.name = "PROPELLER_FAN";
// Position the prop directly at the end of the connecting shaft (facing backward, -Z)
propGroup.position.set(0, 0, -0.24);
outputFlangeGroup.add(propGroup);

// High-Gloss Chrome Spinner Cone (pointing backward, so invert)
const spinnerCone = createZCone(0.26, 0.36, 32, chromeMat);
spinnerCone.rotation.x = Math.PI; // flip to face backward
spinnerCone.position.set(0, 0, -0.16);
spinnerCone.castShadow = true;
propGroup.add(spinnerCone);

const hubPlate = createZCylinder(0.25, 0.25, 0.07, 32, castAluminumMat);
hubPlate.position.set(0, 0, 0.0);
propGroup.add(hubPlate);

// 3 Aero Carbon Fiber Blades with Yellow Safety Tips (angled for pusher prop)
const bladeGeo = new THREE.BoxGeometry(1.65, 0.038, 0.14);
for (let b = 0; b < 3; b++) {
    const bladeHolder = new THREE.Group();
    bladeHolder.rotation.z = (b * 2 * Math.PI) / 3;
    propGroup.add(bladeHolder);

    const bladeMesh = new THREE.Mesh(bladeGeo, propBladeMat);
    bladeMesh.rotation.x = -Math.PI / 8; // reversed pitch for pusher
    bladeMesh.position.set(0.78, 0, -0.05);
    bladeMesh.castShadow = true;
    bladeHolder.add(bladeMesh);

    // Yellow Safety Tip
    const tipMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.18, 0.04, 0.142),
        new THREE.MeshBasicMaterial({ color: 0xfacc15 })
    );
    tipMesh.position.set(1.54, 0, -0.05);
    bladeHolder.add(tipMesh);
}

// ---------------------------------------------------------------------------
// 6. Dynamic Telemetry Data Handling & Color Mapping
// ---------------------------------------------------------------------------
let rpm = 0;
let telemetryLive = false;
let throttleVal = 0.6;
let vibVal = 0.35;
let propAngle = 0;
let crankAngle = 0;
let currentThrottleAngle = 0;
let targetThrottleAngle = 0;
let latestTelemetryData = {};

function tempToColor(val, minTemp, maxTemp) {
    const t = Math.max(0, Math.min(1, (val - minTemp) / (maxTemp - minTemp)));
    let r, g, b;
    if (t < 0.5) {
        const f = t * 2;
        r = Math.floor(255 * f);
        g = 220 + Math.floor(20 * f);
        b = Math.floor(100 * (1 - f));
    } else {
        const f = (t - 0.5) * 2;
        r = 255;
        g = Math.floor(220 * (1 - f));
        b = Math.floor(20 * (1 - f));
    }
    return new THREE.Color(r / 255, g / 255, b / 255);
}

function updateEngineState(data) {
    latestTelemetryData = data;

    // Environmental climate scenery photo update
    const activeProf = data.atmospheric_profile || data.mission || data.climate;
    if (activeProf) {
        updateEnvironmentVisuals(activeProf);
    }

    if (typeof data.rpm === "number") {
        rpm = data.rpm;
    }
    if (typeof data.throttle === "number") {
        throttleVal = data.throttle;
        targetThrottleAngle = Math.max(0, Math.min(1.2, (throttleVal - 0.2) * 1.5));
    }
    if (typeof data.vibration === "number" || typeof data.vibration_measured === "number") {
        vibVal = data.vibration ?? data.vibration_measured ?? 0;
    }

    // Check if Standard Mode / Nominal
    const currentInjectedFault = (data.injected_fault || data.fault_class || data.fault_type || "NONE").toUpperCase();
    const isNominalMode = currentInjectedFault.includes("NONE") || currentInjectedFault.includes("NOMINAL");

    // CHT Thermal Sync (Cylinder Head & Cooling Fins)
    const chtVal = data.cht ?? data.cht_measured;
    if (typeof chtVal === "number" && !isNominalMode) {
        const c = tempToColor(chtVal, 140, 240);
        chtFinsMat.color.copy(c);
        chtFinsMat.emissive.copy(c).multiplyScalar(0.3);
    } else {
        chtFinsMat.color.setHex(0x5a6578);
        chtFinsMat.emissive.setHex(0x000000);
    }

    // EGT Thermal Sync (Exhaust Header Pipe Heat Glow)
    const egtVal = data.egt ?? data.egt_measured;
    if (typeof egtVal === "number" && !isNominalMode) {
        const c = tempToColor(egtVal, 520, 850);
        exhaustHeaderMat.emissive.copy(c);
        const glowFactor = Math.max(0, (egtVal - 600) / 250);
        exhaustHeaderMat.emissiveIntensity = glowFactor * 0.85;
    } else {
        exhaustHeaderMat.emissive.setHex(0x000000);
        exhaustHeaderMat.emissiveIntensity = 0;
    }

    // Oil Temp Thermal Sync (Sump)
    const oilVal = data.oil_temp ?? data.oil_temp_measured;
    if (typeof oilVal === "number" && !isNominalMode) {
        const cOil = tempToColor(oilVal, 70, 125);
        oilPanMat.color.copy(cOil).multiplyScalar(0.4);
    } else {
        oilPanMat.color.setHex(0x1e293b);
    }

    // ---------------------------------------------------------------------------
    // Uniform Component Color Sync (Normal CAD -> Yellow -> Orange -> Red)
    // ---------------------------------------------------------------------------
    const health = typeof data.health_index === "number" ? data.health_index : (typeof data.health === "number" ? data.health : null);
    const healthStatus = (data.health_status || "UNKNOWN").toUpperCase();

    // Extract fault class, anomaly channels, and live telemetry parameter values
    const faultClass = (data.fault_class || data.fault_type || data.fault || "UNAVAILABLE").toUpperCase();
    const anomalyChannels = Array.isArray(data.anomaly_channels) ? data.anomaly_channels : [];

    const liveCht = typeof data.cht === "number" ? data.cht : (typeof data.cht_measured === "number" ? data.cht_measured : null);
    const liveEgt = typeof data.egt === "number" ? data.egt : (typeof data.egt_measured === "number" ? data.egt_measured : null);
    const liveOilTemp = typeof data.oil_temp === "number" ? data.oil_temp : (typeof data.oil_temp_measured === "number" ? data.oil_temp_measured : null);
    const liveOilPress = typeof data.oil_press === "number" ? data.oil_press : (typeof data.oil_press_measured === "number" ? data.oil_press_measured : null);
    const liveMap = typeof data.map === "number" ? data.map : (typeof data.map_measured === "number" ? data.map_measured : null);
    const liveFuelFlow = typeof data.fuel_flow === "number" ? data.fuel_flow : (typeof data.fuel_flow_measured === "number" ? data.fuel_flow_measured : null);

    const expCht = typeof data.cht_expected === "number" ? data.cht_expected : null;
    const expEgt = typeof data.egt_expected === "number" ? data.egt_expected : null;
    const expOilP = typeof data.oil_press_expected === "number" ? data.oil_press_expected : null;
    const expOilT = typeof data.oil_temp_expected === "number" ? data.oil_temp_expected : null;
    const expMap = typeof data.map_expected === "number" ? data.map_expected : null;
    const expFuel = typeof data.fuel_flow_expected === "number" ? data.fuel_flow_expected : null;

    const drift = (a, b, scale) => Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) / scale : 0;

    // Base Nominal Material Colors (Authentic CAD finish at start & nominal mode)
    const BASE_COLORS = {
        castAluminum: 0x5a6578,
        valveCoverRed: 0xd91c1c,
        oilPan: 0x1e293b,
        oilFilterBlue: 0x1d4ed8,
        exhaustHeader: 0x272e3b,
        chargePipe: 0xf8fafc,
        chtFins: 0x00e676
    };

    // Uniform 4-Tier Progression Helper: Normal Base -> Yellow -> Orange -> Red
    function calculateUniformColor(driftRatio, baseNominalHex) {
        if (isNominalMode || driftRatio < 0.15) {
            return baseNominalHex; // Nominal CAD finish (Start & Normal mode)
        } else if (driftRatio < 0.40) {
            return 0xeab308; // Yellow (Mild fault / mild control drift)
        } else if (driftRatio < 0.70) {
            return 0xf97316; // Orange (Moderate fault / moderate control drift)
        } else {
            return 0xef4444; // Red (Severe fault / critical control drift)
        }
    }

    // 1. Engine Block & Cylinder Head (CHT Overheating Drift -> Yellow -> Orange -> Red)
    const chtDriftRatio = Math.max(
        drift(liveCht, expCht, 80.0),
        faultClass.includes("OVERHEAT") ? 0.85 : (anomalyChannels.includes("CHT") ? 0.50 : 0.0)
    );
    castAluminumMat.color.setHex(calculateUniformColor(chtDriftRatio, BASE_COLORS.castAluminum));

    // 2. Oil Pan Sump & Oil Filter (Lubrication Drift -> Yellow -> Orange -> Red)
    const oilDriftRatio = Math.max(
        drift(expOilP, liveOilPress, 25.0),
        drift(liveOilTemp, expOilT, 35.0),
        faultClass.includes("LUBRICATION") || faultClass.includes("OIL") ? 0.85 : 0.0,
        anomalyChannels.includes("OIL_PRESS") || anomalyChannels.includes("OIL_TEMP") ? 0.50 : 0.0
    );
    oilPanMat.color.setHex(calculateUniformColor(oilDriftRatio, BASE_COLORS.oilPan));
    if (oilDriftRatio >= 0.15) {
        oilFilterBlueMat.color.setHex(calculateUniformColor(oilDriftRatio, BASE_COLORS.oilFilterBlue));
    } else {
        oilFilterBlueMat.color.setHex(BASE_COLORS.oilFilterBlue);
    }

    // 3. Exhaust Headers & Volute (EGT / Exhaust Leak Drift -> Yellow -> Orange -> Red)
    const egtDriftRatio = Math.max(
        drift(liveEgt, expEgt, 180.0),
        faultClass.includes("EXHAUST") ? 0.85 : (anomalyChannels.includes("EGT") ? 0.50 : 0.0)
    );
    exhaustHeaderMat.color.setHex(calculateUniformColor(egtDriftRatio, BASE_COLORS.exhaustHeader));
    if (egtDriftRatio >= 0.40) {
        exhaustHeaderMat.emissive.setHex(egtDriftRatio >= 0.70 ? 0xff2200 : 0xaa5500);
        exhaustHeaderMat.emissiveIntensity = 0.9;
    } else {
        const glowFactor = Math.max(0, (liveEgt - 580) / 250);
        const cGlow = tempToColor(liveEgt, 520, 850);
        exhaustHeaderMat.emissive.copy(cGlow);
        exhaustHeaderMat.emissiveIntensity = glowFactor * 0.75;
    }

    // 4. Intake Manifold & Velocity Stack (MAP / Fuel Restriction Drift -> Yellow -> Orange -> Red)
    const fuelDriftRatio = Math.max(
        drift(liveMap, expMap, 14.0),
        drift(expFuel, liveFuelFlow, 4.5),
        faultClass.includes("FUEL") ? 0.85 : 0.0,
        anomalyChannels.includes("FUEL_FLOW") || anomalyChannels.includes("MAP") ? 0.50 : 0.0
    );
    if (typeof chargePipeMat !== "undefined") {
        chargePipeMat.color.setHex(calculateUniformColor(fuelDriftRatio, BASE_COLORS.chargePipe));
    }

    // 5. Valve Cover & Ignition Coils (Misfire / Vibration Drift -> Yellow -> Orange -> Red)
    const misfireDriftRatio = Math.max(
        faultClass.includes("MISFIRE") ? 0.85 : 0.0,
        anomalyChannels.includes("VIB") ? 0.50 : 0.0
    );
    if (misfireDriftRatio >= 0.15) {
        valveCoverRedMat.color.setHex(calculateUniformColor(misfireDriftRatio, BASE_COLORS.valveCoverRed));
    } else {
        valveCoverRedMat.color.setHex(BASE_COLORS.valveCoverRed);
    }

    // Dynamic Atmospheric Background & Visual Effects Sync
    updateEnvironmentVisuals(data.atmospheric_profile);
}



// ---------------------------------------------------------------------------
// 7. 60 FPS Real-Time Animation & Reciprocating Kinematics Loop
// ---------------------------------------------------------------------------
const clock = new THREE.Clock();
let smoothRpm = 0;

function animate() {
    requestAnimationFrame(animate);

    try {
        const rawDelta = clock.getDelta();
        // Clamp delta to 30 FPS step max to prevent giant jumps during frame drops / tab switching
        const delta = Math.min(rawDelta, 0.033);

        if (controls && controls.update) {
            controls.update();
        }

        // Smooth RPM lerp to eliminate sudden speed jerks on throttle changes
        smoothRpm += (rpm - smoothRpm) * 0.08;

        // Responsive smooth visual speed scaling:
        // Low throttle (800 RPM) -> 80 visual RPM (~1.3 revs/sec, smooth lively idle)
        // High throttle (5000 RPM) -> 360 visual RPM (~6 revs/sec, energetic responsive speed)
        const visualRpm = !telemetryLive || rpm <= 0 ? 0 : 80 + Math.min(280, Math.max(0, smoothRpm - 800) * 0.0667);
        const radPerSec = (visualRpm * 2 * Math.PI) / 60;

        propAngle += radPerSec * delta;
        crankAngle += radPerSec * delta;

        // Rotate 3D Forged Steel Crankshaft Assembly
        crankshaftGroup.rotation.z = crankAngle;

        // Flywheel Disc & Rear Power Transmission Reduction Gearbox Kinematics
        flywheelDiscGroup.rotation.z = crankAngle;

        // Gearbox Reduction Drive Kinematics (Meshed Top & Bottom Gears Running)
        // Bottom Drive Gear rotates with Crankshaft (1.0x RPM)
        if (typeof lowerDriveGear !== "undefined") {
            lowerDriveGear.rotation.z = crankAngle;
        }
        // Top Driven Gear rotates in opposite direction with 1.5:1 reduction ratio (20 teeth / 30 teeth = 2/3)
        if (typeof upperDrivenGear !== "undefined") {
            upperDrivenGear.rotation.z = -crankAngle * (20 / 30);
        }
        // Propeller Drive Shaft & Flange spin with Top Driven Gear
        if (typeof outputFlangeGroup !== "undefined" && typeof upperDrivenGear !== "undefined") {
            outputFlangeGroup.rotation.z = upperDrivenGear.rotation.z;
        }

        // Turbo Impeller Wheel Rotation
        impellerGroup.rotation.z = crankAngle * 2.5;

        // Serpentine Drive Pulleys & Alternator Fan Rotation
        crankPulley.rotation.z = crankAngle;
        altPulley.rotation.z = -crankAngle * 1.5;
        altFanMesh.rotation.z = -crankAngle * 1.5;
        wpPulley.rotation.z = crankAngle * 1.25;
        tensionerPulley.rotation.z = -crankAngle * 1.8;

        // Rotate DOHC Twin Camshaft Assemblies at 1:2 speed relative to crankshaft
        intakeCamshaftGroup.rotation.z = crankAngle * 0.5;
        exhaustCamshaftGroup.rotation.z = crankAngle * 0.5;

        // Rotate DOHC Camshaft Gears & Crankshaft Timing Sprocket in exact 1:2 mechanical ratio
        leftCamGear.rotation.z = crankAngle * 0.5;
        rightCamGear.rotation.z = crankAngle * 0.5;
        crankTimingGear.rotation.z = crankAngle;

        // Continuously advance timing chain links along the closed 3D timing path in real engine motion
        if (typeof timingChainCurve !== "undefined" && typeof timingChainLinks !== "undefined" && timingChainLinks.length > 0) {
            const chainTotalLen = timingChainCurve.getLength();
            const chainDist = crankAngle * 0.10;
            const chainOffset = (chainDist / chainTotalLen) % 1.0;

            for (let k = 0; k < timingChainLinks.length; k++) {
                const link = timingChainLinks[k];
                let t = (k / timingChainLinks.length + chainOffset) % 1.0;
                if (t < 0) t += 1.0;

                const pt = timingChainCurve.getPointAt(t);
                const tan = timingChainCurve.getTangentAt(t);

                link.position.set(pt.x, pt.y, pt.z);
                link.rotation.z = Math.atan2(tan.y, tan.x);
            }
        }

        // Rocker Arm Pivoting Kinematics (Pivot on rocker shafts as cam lobes depress poppet valves)
        for (let r = 0; r < rockerArms.length; r++) {
            const rObj = rockerArms[r];
            const i = rObj.cylIndex;
            const fireOffset = firingPhaseOffsets[i] || 0;
            const cycleAngle = ((crankAngle + fireOffset) % (4 * Math.PI) + 4 * Math.PI) % (4 * Math.PI);

            let lift = 0;
            if (rObj.type === "INTAKE" && cycleAngle < Math.PI) {
                lift = 0.038 * Math.sin(cycleAngle);
            } else if (rObj.type === "EXHAUST" && cycleAngle >= 3 * Math.PI) {
                lift = 0.038 * Math.sin(cycleAngle - 3 * Math.PI);
            }

            const pivotTilt = (rObj.type === "INTAKE" ? -1 : 1) * (lift / 0.038) * 0.28;
            rObj.mesh.rotation.z = pivotTilt;
        }

        // Throttle Lever Smooth Movement
        currentThrottleAngle += (targetThrottleAngle - currentThrottleAngle) * 0.1;
        throttleLever.rotation.z = currentThrottleAngle;

        // Inline-4 Reciprocating Pistons & Connecting Rod Kinematics (Exact Pinned Slider-Crank)
        const strokeRadius = 0.095;
        const rodLength = 0.32;

        for (let i = 0; i < pistons.length; i++) {
            const pObj = pistons[i];
            const rObj = conRods[i];
            const currentCrank = crankAngle + pObj.phase;

            // Global Rotating Crankpin Center Position for cylinder i
            const pinX = Math.cos(currentCrank) * strokeRadius;
            const pinY = 0.52 + Math.sin(currentCrank) * strokeRadius;

            // Exact Slider-Crank Piston Y Position (Wrist pin at X = 0, Y = pistonY)
            const dyRod = Math.sqrt(Math.max(0, rodLength * rodLength - pinX * pinX));
            const pistonY = pinY + dyRod;

            // Piston Crown Assembly Y Position
            pObj.mesh.position.y = pistonY;

            // Connecting Rod Position (Center of segment between Wrist Pin (0, pistonY) and Crankpin (pinX, pinY))
            const rodCenterX = pinX * 0.5;
            const rodCenterY = (pinY + pistonY) * 0.5;
            rObj.mesh.position.set(rodCenterX, rodCenterY, pObj.baseZ);

            // Connecting Rod Angular Tilt (Pivots at Wrist Pin and encircles Crankpin)
            const rodTilt = Math.atan2(pinX, pistonY - pinY);
            rObj.mesh.rotation.z = rodTilt;

            // 4-Stroke Cycle angle for cylinder i (spans 4*PI radians = 720 degrees)
            const fireOffset = firingPhaseOffsets[i] || 0;
            const cycleAngle = ((crankAngle + fireOffset) % (4 * Math.PI) + 4 * Math.PI) % (4 * Math.PI);

            // -------------------------------------------------------------------
            // DOHC POPPET VALVE MOVEMENTS (Intake & Exhaust Lift Kinematics)
            // -------------------------------------------------------------------
            let intakeLift = 0;
            let exhaustLift = 0;

            if (cycleAngle < Math.PI) {
                // Intake Stroke (0 -> 180 deg): Intake valves open downward
                intakeLift = 0.038 * Math.sin(cycleAngle);
            } else if (cycleAngle >= 3 * Math.PI) {
                // Exhaust Stroke (540 -> 720 deg): Exhaust valves open downward
                exhaustLift = 0.038 * Math.sin(cycleAngle - 3 * Math.PI);
            }

            const cylValves = valvesByCylinder[i];
            if (cylValves) {
                for (let v = 0; v < cylValves.length; v++) {
                    const vObj = cylValves[v];
                    if (vObj.type === "INTAKE") {
                        vObj.mesh.position.y = vObj.baseY - intakeLift;
                    } else if (vObj.type === "EXHAUST") {
                        vObj.mesh.position.y = vObj.baseY - exhaustLift;
                    }
                }
            }

            // -------------------------------------------------------------------
            // SPARK PLUG HIGH-VOLTAGE IGNITION ARC FLASH (At Top Dead Center = 360 deg)
            // -------------------------------------------------------------------
            if (sparkArcs[i]) {
                const sObj = sparkArcs[i];
                // Spark fires instantly at 360 deg (2*PI) to 405 deg (2.25*PI)
                const isSparking = (cycleAngle >= 2 * Math.PI && cycleAngle < 2.25 * Math.PI);
                sObj.arcMesh.visible = isSparking;
                sObj.coreMesh.visible = isSparking;
                if (isSparking) {
                    const sparkPulse = 1.0 + Math.sin(clock.elapsedTime * 60) * 0.3;
                    sObj.arcMesh.scale.setScalar(sparkPulse);
                }
            }

            // -------------------------------------------------------------------
            // 4-STROKE DUAL-LAYER COMBUSTION CHAMBER FLAME & EXPLOSION VISUALIZATION
            // -------------------------------------------------------------------
            if (combustionChambers[i]) {
                const cObj = combustionChambers[i];

                // Calculate dynamic chamber height and center Y above piston crown
                const crownY = pistonY + 0.08;
                const deckY = 1.15;
                const chamberHeight = Math.max(0.025, deckY - crownY);
                cObj.mesh.position.y = crownY + chamberHeight / 2;
                cObj.mesh.scale.y = chamberHeight / 0.20;

                if (cObj.coreMesh) {
                    cObj.coreMesh.position.y = crownY + chamberHeight / 2;
                    cObj.coreMesh.scale.y = chamberHeight / 0.20;
                }

                // Color & Ignition Flash based on 4-stroke cycle phase
                if (cycleAngle < Math.PI) {
                    // 1. INTAKE STROKE (0 -> 180 deg): Cool Air-Fuel Swirl Charge (Cyan / Sky Blue)
                    const t = cycleAngle / Math.PI;
                    cObj.mat.color.setHex(0x00d2ff);
                    cObj.mat.emissive.setHex(0x00a2ff);
                    cObj.mat.emissiveIntensity = 1.2 + 0.6 * Math.sin(t * Math.PI);
                    cObj.mat.opacity = 0.40 + 0.35 * Math.sin(t * Math.PI);
                    if (cObj.coreMesh) cObj.coreMesh.visible = false;
                } else if (cycleAngle < 2 * Math.PI) {
                    // 2. COMPRESSION STROKE (180 -> 360 deg): High-Density Heat (Golden Yellow)
                    const t = (cycleAngle - Math.PI) / Math.PI;
                    cObj.mat.color.setHex(0xffb300);
                    cObj.mat.emissive.setHex(0xff8800);
                    cObj.mat.emissiveIntensity = 1.5 + 1.2 * t;
                    cObj.mat.opacity = 0.50 + 0.40 * t;
                    if (cObj.coreMesh) cObj.coreMesh.visible = false;
                } else if (cycleAngle < 3 * Math.PI) {
                    // 3. POWER / COMBUSTION STROKE (360 -> 540 deg): Spark Ignition & White-Hot Flame Burst!
                    const t = (cycleAngle - 2 * Math.PI) / Math.PI;
                    if (t < 0.25) {
                        // Instant Spark Plug Explosion Burst (Crimson Flame with White-Hot Center Core)
                        cObj.mat.color.setHex(0xffffff);
                        cObj.mat.emissive.setHex(0xff2200);
                        cObj.mat.emissiveIntensity = 5.0;
                        cObj.mat.opacity = 0.98;
                        if (cObj.coreMesh) {
                            cObj.coreMesh.visible = true;
                            cObj.coreMat.emissiveIntensity = 6.0;
                            cObj.coreMat.opacity = 0.95;
                        }
                    } else {
                        // Expanding Flame Front Body
                        cObj.mat.color.setHex(0xef4444);
                        cObj.mat.emissive.setHex(0xd97706);
                        cObj.mat.emissiveIntensity = 3.5 * Math.exp(-t * 2.0) + 1.0;
                        cObj.mat.opacity = 0.88 * Math.exp(-t * 1.5) + 0.30;
                        if (cObj.coreMesh) {
                            cObj.coreMesh.visible = true;
                            cObj.coreMat.emissiveIntensity = 3.0 * (1.0 - t);
                            cObj.coreMat.opacity = 0.60 * (1.0 - t);
                        }
                    }
                } else {
                    // 4. EXHAUST STROKE (540 -> 720 deg): Spent Hot Exhaust Evacuation (Smoky Orange / Amber)
                    const t = (cycleAngle - 3 * Math.PI) / Math.PI;
                    cObj.mat.color.setHex(0xf97316);
                    cObj.mat.emissive.setHex(0xb91c1c);
                    cObj.mat.emissiveIntensity = 1.0 * (1.0 - t);
                    cObj.mat.opacity = 0.45 * (1.0 - t);
                    if (cObj.coreMesh) cObj.coreMesh.visible = false;
                }
            }
        }

        // Structural Micro-Vibration Jitter (Smooth, subtle operational jitter)
        const vibJitterScale = vibVal > 1.8
            ? Math.min(0.005, (vibVal - 0.2) * 0.001)
            : Math.min(0.002, Math.max(0, vibVal - 0.35) * 0.0006);

        if (vibJitterScale > 0.0001) {
            engineMasterGroup.position.x = (Math.random() - 0.5) * vibJitterScale;
            engineMasterGroup.position.y = (Math.random() - 0.5) * vibJitterScale;
            engineMasterGroup.position.z = (Math.random() - 0.5) * vibJitterScale;
        } else {
            engineMasterGroup.position.set(0, 0, 0);
        }

        // Environmental Particle Dynamics Animation
        if (particleMat.opacity > 0.01) {
            const posAttr = particleGeo.attributes.position;
            const posArr = posAttr.array;
            for (let i = 0; i < particleCount; i++) {
                const idx = i * 3;
                if (currentEnvProfile === "ARCTIC_FREEZE") {
                    // Cold snow flurry drifting downward and tumbling
                    posArr[idx + 1] -= (0.04 + Math.random() * 0.02);
                    posArr[idx] += Math.sin(clock.elapsedTime * 2.0 + i) * 0.008;
                    if (posArr[idx + 1] < -2.0) {
                        posArr[idx + 1] = 10.0;
                        posArr[idx] = (Math.random() - 0.5) * 20;
                    }
                } else if (currentEnvProfile === "LIVE_STORM_TURBULENCE") {
                    // Rain / tempest wind streaks slanting across screen
                    posArr[idx + 1] -= (0.14 + Math.random() * 0.05);
                    posArr[idx] -= (0.05 + Math.random() * 0.02);
                    if (posArr[idx + 1] < -2.0 || posArr[idx] < -10.0) {
                        posArr[idx + 1] = 10.0;
                        posArr[idx] = (Math.random() - 0.5) * 20 + 5.0;
                    }
                } else if (currentEnvProfile === "HIGH_HEAT_DESERT") {
                    // Thermal heat embers/dust floating upward
                    posArr[idx + 1] += 0.015;
                    posArr[idx] += Math.sin(clock.elapsedTime * 1.5 + i) * 0.006;
                    if (posArr[idx + 1] > 10.0) {
                        posArr[idx + 1] = -2.0;
                        posArr[idx] = (Math.random() - 0.5) * 20;
                    }
                } else if (currentEnvProfile === "HIGH_ALTITUDE_THIN_AIR") {
                    // High altitude ice crystals hovering smoothly
                    posArr[idx] += Math.cos(clock.elapsedTime * 0.3 + i) * 0.002;
                    posArr[idx + 1] += Math.sin(clock.elapsedTime * 0.3 + i) * 0.002;
                } else {
                    // ISA_STANDARD: subtle floating atmospheric sparkles
                    posArr[idx + 1] += Math.sin(clock.elapsedTime * 0.5 + i) * 0.001;
                    posArr[idx] += Math.cos(clock.elapsedTime * 0.5 + i) * 0.001;
                }
            }
            posAttr.needsUpdate = true;

            if (currentEnvProfile === "LIVE_STORM_TURBULENCE") {
                stormFlashTimer += delta;
                if (stormFlashTimer > 2.8 && Math.random() < 0.12) {
                    ambientLight.intensity = 2.2;
                    setTimeout(() => { ambientLight.intensity = 0.75; }, 70);
                    stormFlashTimer = 0.0;
                }
            }
        }

        renderer.render(scene, camera);
    } catch (err) {
        console.error("viewer.js animation error:", err);
    }
}

animate();

// ---------------------------------------------------------------------------
// 8. Raycaster & Interactive Component Inspection HUD Overlay
// ---------------------------------------------------------------------------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function findNamedParent(obj) {
    let curr = obj;
    while (curr) {
        if (curr.name && curr.name !== "") {
            return curr.name;
        }
        curr = curr.parent;
    }
    return "ENGINE_ASSEMBLY";
}

window.addEventListener("pointerdown", (event) => {
    if (event.target.closest("#info, #inspector-hud")) return;
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(engineMasterGroup.children, true);

    if (intersects.length > 0) {
        const hit = intersects[0].object;
        const compName = findNamedParent(hit);
        const hud = document.getElementById("inspector-hud");
        const titleEl = document.getElementById("hud-title");
        const contentEl = document.getElementById("hud-content");

        if (hud && titleEl && contentEl) {
            hud.style.display = "block";
            const egt = latestTelemetryData.egt ?? latestTelemetryData.egt_measured ?? "Unavailable";
            const cht = latestTelemetryData.cht ?? latestTelemetryData.cht_measured ?? "Unavailable";
            const egtExp = latestTelemetryData.egt_expected ?? "Unavailable";
            const chtExp = latestTelemetryData.cht_expected ?? "Unavailable";
            const oilP = latestTelemetryData.oil_press ?? latestTelemetryData.oil_press_measured ?? "Unavailable";
            const oilT = latestTelemetryData.oil_temp ?? latestTelemetryData.oil_temp_measured ?? "Unavailable";
            const mapV = latestTelemetryData.map ?? latestTelemetryData.map_measured ?? "Unavailable";
            const vibV = latestTelemetryData.vibration ?? latestTelemetryData.vibration_measured ?? "Unavailable";
            const health = latestTelemetryData.health_status || "UNKNOWN";
            const activeFault = (latestTelemetryData.fault_class || latestTelemetryData.fault_type || "UNAVAILABLE").toUpperCase();
            const anomalyActive = latestTelemetryData.anomaly_detected === true || (typeof latestTelemetryData.health_index === "number" && latestTelemetryData.health_index < 80);

            let statusTagColor = "#38bdf8"; // cyan nominal
            if (anomalyActive) {
                if (health === "CRITICAL" || (typeof latestTelemetryData.health_index === "number" && latestTelemetryData.health_index < 40)) {
                    statusTagColor = "#ef4444"; // red
                } else if (health === "WARNING" || (typeof latestTelemetryData.health_index === "number" && latestTelemetryData.health_index < 60)) {
                    statusTagColor = "#f97316"; // orange
                } else {
                    statusTagColor = "#f59e0b"; // yellow
                }
            }

            if (compName.includes("VALVE_COVER")) {
                titleEl.innerText = "🔴 Organic Red DOHC Valve Cover & Ignition Coils";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Dual Overhead Cam Valve Cover<br>
                    <b>Ignition Coils:</b> 4 Active Coil-on-Plug Packs<br>
                    <b>Cylinder Head Temp:</b> ${cht} °C (Target: ${chtExp} °C)<br>
                    <b>Active Fault:</b> ${activeFault !== "NONE" ? activeFault : "None"}<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor}; font-weight:600;">Status: ${health} (${anomalyActive ? "Color Sync: Active Fault Alert" : "Normal CAD Finish"})</div>
                `;
            } else if (compName.includes("CYLINDER_HEAD")) {
                titleEl.innerText = "⚙️ Sculpted Contoured Cylinder Block Structure";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Inline-4 Contoured Cylinder Barrels & Deck<br>
                    <b>Measured CHT:</b> ${cht} °C<br>
                    <b>Physics Target:</b> ${chtExp} °C<br>
                    <b>CHT Residual Drift:</b> ${latestTelemetryData.res_cht ?? "Unavailable"} °C<br>
                    <b>Active Fault:</b> ${activeFault !== "NONE" ? activeFault : "None"}<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor}; font-weight:600;">Status: ${health} (${anomalyActive && activeFault.includes("OVERHEAT") ? "Block Color: Thermal Warning Sync" : "Normal Material"})</div>
                `;
            } else if (compName.includes("EXHAUST_HEADERS")) {
                titleEl.innerText = "🌀 360° Volute Turbocharger & Swept Headers";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Full 360° Ring Turbo Volute & Swept Manifold<br>
                    <b>Measured EGT:</b> ${egt} °C<br>
                    <b>Physics Target:</b> ${egtExp} °C<br>
                    <b>Boost Pressure (MAP):</b> ${mapV} inHg<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor}; font-weight:600;">Status: ${health} (Exhaust Header Glow: Sync)</div>
                `;
            } else if (compName.includes("TIMING_CHAIN")) {
                titleEl.innerText = "⛓️ DOHC Timing Chain & Blue Guide Rail";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Extruded Camshaft Gears & Blue Guide Rail<br>
                    <b>Timing Ratio:</b> 1:2 Crankshaft Speed<br>
                    <b>Engine Speed:</b> ${rpm.toFixed(0)} RPM<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor};">Status: ${health}</div>
                `;
            } else if (compName.includes("ALTERNATOR_SERPENTINE")) {
                titleEl.innerText = "⚡ Slotted Alternator with Pulley Fan & Serpentine Belt";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Slotted Alternator, Pulley Fan & Coplanar Belt<br>
                    <b>Ventilation:</b> 12 Body Slot Lines & Small Cooling Impeller Fan<br>
                    <b>Belt Speed:</b> Synchronized to ${rpm.toFixed(0)} RPM<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor};">Status: ${health}</div>
                `;
            } else if (compName.includes("INTAKE_THROTTLE")) {
                titleEl.innerText = "🌀 Intake Manifold with Open Front Velocity Stack";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Aluminum Plenum Log & Open Velocity Stack<br>
                    <b>Manifold Press (MAP):</b> ${mapV} inHg<br>
                    <b>Throttle Position:</b> ${(throttleVal * 100).toFixed(1)} %<br>
                    <b>Intake Air Temp (IAT):</b> ${latestTelemetryData.iat || "—"} °C<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor}; font-weight:600;">Status: ${health} (${activeFault.includes("FUEL") ? "Fuel Restriction Color Sync: Active" : "Nominal"})</div>
                `;
            } else if (compName.includes("OIL_SUMP_FILTER")) {
                titleEl.innerText = "🛢️ Stepped Wet Sump Oil Pan & Blue Oil Filter";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Deep Sump Well & 16-Bolt Sealing Flange<br>
                    <b>Oil Pressure:</b> ${oilP} PSI<br>
                    <b>Oil Temperature:</b> ${oilT} °C<br>
                    <b>Pressure Target:</b> ${latestTelemetryData.oil_press_expected || "—"} PSI<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor}; font-weight:600;">Status: ${health} (${activeFault.includes("LUBRICATION") || activeFault.includes("OIL") ? "Lubrication Alert Color Sync: Active" : "Nominal"})</div>
                `;
            } else if (compName.includes("REAR_FLYWHEEL")) {
                titleEl.innerText = "⚙️ Gearbox Transmission & Rotating Output Flange";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Power Transmission Gearbox & Rotating Flange<br>
                    <b>Rotating Parts:</b> Flywheel & Output Coupling Flange<br>
                    <b>Transmission Speed:</b> ${rpm.toFixed(0)} RPM<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor};">Power Output: Active (${health})</div>
                `;
            } else if (compName.includes("PROPELLER_FAN")) {
                titleEl.innerText = "✈️ Aero Propeller Assembly";
                contentEl.innerHTML = `
                    <b>Sub-Assembly:</b> Chrome Spinner Cone & Carbon Fiber Blades<br>
                    <b>Rotational Speed:</b> ${rpm.toFixed(0)} RPM<br>
                    <b>Vibration Level:</b> ${vibV} g<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor};">Status: ${health}</div>
                `;
            } else {
                titleEl.innerText = "🔍 Inline-4 Turbocharged Aero Engine Twin";
                contentEl.innerHTML = `
                    <b>Measured EGT:</b> ${egt} °C | <b>CHT:</b> ${cht} °C<br>
                    <b>Engine Speed:</b> ${rpm.toFixed(0)} RPM<br>
                    <b>Structural Vib:</b> ${vibV} g<br>
                    <b>Active Fault Class:</b> ${activeFault !== "NONE" ? activeFault : "NOMINAL"}<br>
                    <div style="margin-top:6px; font-size:11px; color:${statusTagColor}; font-weight:600;">Engine risk state: ${health}</div>
                `;
            }
        }
    }
});

// React owns the API connection and supplies the same snapshot shown in its tables.
window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin || event.source !== window.parent) return;
    if (event.data?.type === "telemetry" && event.data.payload) {
        telemetryLive = event.data.live === true;
        updateEngineState(event.data.payload);
        if (!event.data.live || !Number.isFinite(event.data.payload.rpm)) rpm = 0;
        const status = document.getElementById("telemetry-status");
        if (status) status.textContent = event.data.live ? `LIVE · ${Number(event.data.payload.rpm || 0).toFixed(0)} RPM · T+ ${event.data.payload.timestamp}s` : "TELEMETRY STALE · ANIMATION PAUSED";
    }
});

// Canvas Auto-Resize
function resizeCanvasIfNeeded() {
    const width = document.documentElement.clientWidth || window.innerWidth;
    const height = document.documentElement.clientHeight || window.innerHeight;
    if (width > 50 && height > 50) {
        const targetPixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        const currentW = renderer.domElement.width;
        const currentH = renderer.domElement.height;
        const expectedW = Math.floor(width * targetPixelRatio);
        const expectedH = Math.floor(height * targetPixelRatio);

        if (Math.abs(currentW - expectedW) > 2 || Math.abs(currentH - expectedH) > 2) {
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height, false);
        }
    }
}

window.addEventListener("resize", resizeCanvasIfNeeded);

if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => {
        resizeCanvasIfNeeded();
    });
    ro.observe(document.body);
    if (document.documentElement) {
        ro.observe(document.documentElement);
    }
}

setTimeout(resizeCanvasIfNeeded, 100);
setTimeout(resizeCanvasIfNeeded, 500);