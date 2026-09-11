// I hope you hate your GPU as much as I do

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>

#include <switch.h>
#include <deko3d.h>

#ifndef DK_UNIFORM_BUF_ALIGNMENT
#define DK_UNIFORM_BUF_ALIGNMENT 0x100
#endif

#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((uint32_t)(a) - 1))

#define WG_SIZE_X 16
#define WG_SIZE_Y 16
// groups are dispatched in a 4x4 grid, for 2048 threads total
// 2048 per SM = 4096 on TX1, so we fully saturate the GPU
#define NUM_GROUPS_X 4
#define NUM_GROUPS_Y 4

#define N_THREADS ((NUM_GROUPS_X * WG_SIZE_X) * (NUM_GROUPS_Y * WG_SIZE_Y))  // 4096
#define N_ROWS 64

// Thread buffers = N_ROWS * N_THREADS, must be power of 2
#define N_VEC ((uint64_t)N_ROWS * N_THREADS)   // uvec4 elements, 2^18
#define N_ELEMS (N_VEC * 4)                      // uints, 2^20

_Static_assert((N_VEC & (N_VEC - 1)) == 0,
               "N_ROWS * N_THREADS must be a power of two for the scatter mask");

// Gather size, 8192 * 16B = 128KB per SM. Fits into L2 perfectly
#define L2_WINDOW_VEC 8192u                      
#define DISPATCHES_PER_LIST 32
#define NUM_INFLIGHT 3
#define MAX_RECORDS 64          // fault records per batch, per slot

#define FENCE_TIMEOUT_NS (5ULL * 1000000000ULL)
#define DATA_CHECK_INTERVAL 512

#define CODEMEM_SIZE (128 * 1024)
#define CMDMEM_SIZE (512 * 1024)

#define DKSH_MAGIC UINT32_C(0x48534B44)

// Dispaly setup
#define SCR_COLS 79
#define LOG_LINES 14

#define ROW_TITLE 1
#define ROW_STATS 3    // .. ROW_STATS+3
#define ROW_ERRHDR 8
#define ROW_ERRORS 9    // .. ROW_ERRORS+5
#define ROW_LOGHDR 16
#define ROW_LOG 17   // .. ROW_LOG+LOG_LINES-1
#define ROW_HELP (ROW_LOG + LOG_LINES + 1)   // 32
#define ROW_STATUS (ROW_HELP + 2)              // 34

typedef struct {
    uint32_t magic;
    uint32_t header_sz;
    uint32_t control_sz;
    uint32_t code_sz;
    uint32_t programs_off;
    uint32_t num_programs;
} DkshHeaderLocal;

// Shader parameters
typedef struct {
    uint32_t n_rows;
    uint32_t n_repeats;
    uint32_t alu_iters;
    uint32_t gather_iters;
    uint32_t max_records;
    uint32_t mode;          // 0 = capture golden, 1 = compare, 2 = ignore
    uint32_t scatter_mask;  // Gather window - 1, used for errors.
    uint32_t chain_bias;    
} Params;

// Result buffer
typedef struct {
    uint32_t mismatches;
    uint32_t digest;
    uint32_t selfFail;
    uint32_t errCount;
} GpuResult;

// Errorlog
typedef struct {
    uint32_t id;            // thread index
    uint32_t stage;         // 1 = ALU self-check, 2 = per-thread digest
    uint32_t got;
    uint32_t want;
} GpuRecord;


// Presets for calibration
typedef struct {
    const char* name;
    uint32_t    alu;
    uint32_t    gather;
    uint32_t    window;     // 0 = whole buffer
} Profile;

// Testing modes
// Balanced tried to hit both
// Core forces 2x chains and not as much memory access, although in practice it strains it enough.
// Dram scatters reads accross allocated size (4MB, lol, lmao even)
// L2 changes this to 256KB.
static const Profile kProfiles[] = {
    { "balanced", 64, 32, 0 },
    { "core", 512, 4, 0 },  
    { "dram", 8, 256, 0 },  
    { "l2", 8, 256, L2_WINDOW_VEC },  
};
#define NUM_PROFILES ((int)(sizeof(kProfiles) / sizeof(kProfiles[0])))

static uint32_t scatterMaskFor(int profile)
{
    uint32_t win = kProfiles[profile].window;
    if (win == 0u) win = (uint32_t)N_VEC;
    return win - 1u;
}

// Error storage
typedef struct {
    uint64_t batches;
    uint64_t detected;      // batches where digest=/count
    uint64_t transient;     // corrected by the re-run vote
    uint64_t persistent;    // reproduced on re-run
    uint64_t selfFail;      // ALU DUP/COMP Fails
    uint64_t dataBad;       // corrupt inputs (how did this happen)
    uint64_t warnings;
    uint64_t timeouts;
} Tally;
// A long time ago, and also now, and also never
// nothing was nowhere
// makes sense right?
// Like I said it didn't happen
// Its so every you don't even need a where
// You don't even need a when
// thats how every it gets

static Tally g_tally;
static uint64_t g_tStart;
static bool g_hud;

static char g_log[LOG_LINES][SCR_COLS + 1];
static uint32_t g_logCount;


static uint64_t elapsedSec(void)
{
    return armTicksToNs(armGetSystemTick() - g_tStart) / 1000000000ULL;
}

__attribute__((format(printf, 2, 3)))
static void drawAt(int row, const char* fmt, ...)
{
    char buf[SCR_COLS + 1];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    printf("\x1b[%d;1H%-*s", row, SCR_COLS, buf);
}

__attribute__((format(printf, 1, 2)))
static void note(const char* fmt, ...)
{
    char body[SCR_COLS + 1];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);

    if (!g_hud) {
        printf("%s\n", body);
        consoleUpdate(NULL);
        return;
    }

    uint32_t idx = g_logCount % LOG_LINES;
    snprintf(g_log[idx], sizeof(g_log[idx]), "[%5llus] %s",
             (unsigned long long)elapsedSec(), body);
    g_logCount++;
}

static void drawLog(void)
{
    uint32_t shown = (g_logCount < LOG_LINES) ? g_logCount : LOG_LINES;
    for (uint32_t i = 0; i < LOG_LINES; i++) {
        if (i < shown) {
            uint32_t idx = (g_logCount - 1 - i) % LOG_LINES;
            drawAt(ROW_LOG + (int)i, "%s", g_log[idx]);
        } else {
            drawAt(ROW_LOG + (int)i, " ");
        }
    }
}

static void drawTally(void)
{
    drawAt(ROW_ERRHDR, "ERRORS (cumulative)");
    drawAt(ROW_ERRORS + 0, "  digest mismatches    : %llu",
           (unsigned long long)g_tally.detected);
    drawAt(ROW_ERRORS + 1, "  corrected (transient) : %llu",
           (unsigned long long)g_tally.transient);
    drawAt(ROW_ERRORS + 2, "  persistent            : %llu",
           (unsigned long long)g_tally.persistent);
    drawAt(ROW_ERRORS + 3, "  ALU self-check fails  : %llu",
           (unsigned long long)g_tally.selfFail);
    drawAt(ROW_ERRORS + 4, "  input data corrupt    : %llu",
           (unsigned long long)g_tally.dataBad);
    drawAt(ROW_ERRORS + 5, "  timeouts / warnings   : %llu / %llu",
           (unsigned long long)g_tally.timeouts,
           (unsigned long long)g_tally.warnings);
}


// Generate seeds
static uint32_t hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}


// RNG, clamped so nothing fucking explodes too hard
static uint32_t dataBits(uint64_t index)
{
    uint32_t bits = hash32((uint32_t)index);
    uint32_t exp  = (bits >> 23) & 0xFFu;
    if (exp < 0x30u) exp = 0x30u + (exp & 0x0Fu);
    if (exp > 0xC0u) exp = 0xC0u - (exp & 0x0Fu);
    return (bits & 0x807FFFFFu) | (exp << 23);
}


typedef struct {
    DkMemBlock block;
    uint32_t offset;
    uint32_t size;
} BumpAlloc;

static uint32_t bumpAlloc(BumpAlloc* a, uint32_t size, uint32_t align)
{
    uint32_t off = ALIGN_UP(a->offset, align);
    if (off + size > a->size) {
        note("out of memory (need %u, have %u)", off + size, a->size);
        return UINT32_MAX;
    }
    a->offset = off + size;
    return off;
}

static void* cpuPtr(BumpAlloc* a, uint32_t off)
{
    return (uint8_t*)dkMemBlockGetCpuAddr(a->block) + off;
}

static DkGpuAddr gpuPtr(BumpAlloc* a, uint32_t off)
{
    return dkMemBlockGetGpuAddr(a->block) + off;
}

static void debugCallback(void* userData, const char* context,
                          DkResult result, const char* message)
{
    (void)userData;
    if (result == DkResult_Success) {
        g_tally.warnings++;
        note("deko3d warn [%s]: %s", context, message);
        return;
    }
    note("deko3d ERROR [%s]: %s", context, message);
    if (g_hud) { drawLog(); drawTally(); }
    consoleUpdate(NULL);
    svcSleepThread(10ULL * 1000000000ULL);
    exit(EXIT_FAILURE);
}

static bool loadShader(DkShader* shader, BumpAlloc* codeMem, const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { note("cannot open %s", path); return false; }

    DkshHeaderLocal hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != DKSH_MAGIC) {
        note("%s is not a DKSH file", path);
        fclose(f);
        return false;
    }

    void* control = malloc(hdr.control_sz);
    if (!control) { fclose(f); return false; }

    rewind(f);
    if (fread(control, hdr.control_sz, 1, f) != 1) {
        free(control); fclose(f); return false;
    }

    uint32_t codeOff = bumpAlloc(codeMem, hdr.code_sz, DK_SHADER_CODE_ALIGNMENT);
    if (codeOff == UINT32_MAX) { free(control); fclose(f); return false; }

    fseek(f, hdr.control_sz, SEEK_SET);
    if (fread(cpuPtr(codeMem, codeOff), hdr.code_sz, 1, f) != 1) {
        free(control); fclose(f); return false;
    }
    fclose(f);

    DkShaderMaker maker;
    dkShaderMakerDefaults(&maker, codeMem->block, codeOff);
    maker.control = control;
    dkShaderInitialize(shader, &maker);

    free(control);
    return dkShaderIsValid(shader);
}

// Usual Deko3D memory allocation
typedef struct {
    DkQueue queue;
    DkCmdBuf cmdbuf;
    DkShader shader;

    DkGpuAddr uboAddr;
    uint32_t uboSize;

    //3x slots, two are constantly worked on while a third is checked
    DkGpuAddr resultAddr[NUM_INFLIGHT];
    GpuResult* resultCpu[NUM_INFLIGHT];
    DkGpuAddr recAddr[NUM_INFLIGHT];
    GpuRecord* recCpu[NUM_INFLIGHT];
    DkCmdList batch[NUM_INFLIGHT];

    DkGpuAddr dataAddr;
    uint32_t dataSize;
    uint32_t* dataCpu;

    // Golden digests
    DkGpuAddr expAddr;    
    uint32_t expSize;
} Ctx;

static void recordBatches(Ctx* c, Params params, uint32_t dispatches)
{
    dkCmdBufClear(c->cmdbuf);

    for (uint32_t s = 0; s < NUM_INFLIGHT; s++) {
        const DkShader* shaders[] = { &c->shader };
        dkCmdBufBindShaders(c->cmdbuf, DkStageFlag_Compute, shaders, 1);

        DkBufExtents ubo = { c->uboAddr, c->uboSize };
        dkCmdBufBindUniformBuffers(c->cmdbuf, DkStage_Compute, 0, &ubo, 1);
        dkCmdBufPushConstants(c->cmdbuf, c->uboAddr, c->uboSize,
                              0, sizeof(params), &params);

        DkBufExtents ssbos[4];
        ssbos[0].addr = c->dataAddr;      ssbos[0].size = c->dataSize;
        ssbos[1].addr = c->resultAddr[s]; ssbos[1].size = sizeof(GpuResult);
        ssbos[2].addr = c->recAddr[s];    ssbos[2].size = MAX_RECORDS * sizeof(GpuRecord);
        ssbos[3].addr = c->expAddr;       ssbos[3].size = c->expSize;
        dkCmdBufBindStorageBuffers(c->cmdbuf, DkStage_Compute, 0, ssbos, 4);

        for (uint32_t i = 0; i < dispatches; i++)
            dkCmdBufDispatchCompute(c->cmdbuf, NUM_GROUPS_X, NUM_GROUPS_Y, 1);

        c->batch[s] = dkCmdBufFinishList(c->cmdbuf);
    }
}

typedef enum { RUN_OK = 0, RUN_TIMEOUT, RUN_QUEUE_ERROR } RunStatus;

static RunStatus runSync(Ctx* c, uint32_t slot, GpuResult* out, uint64_t* outNs)
{
    memset(c->resultCpu[slot], 0, sizeof(GpuResult));

    DkFence fence;
    memset(&fence, 0, sizeof(fence));

    uint64_t t0 = armGetSystemTick();
    dkQueueSubmitCommands(c->queue, c->batch[slot]);
    // CPU politely asks for the data
    dkQueueSignalFence(c->queue, &fence, true);  
    dkQueueFlush(c->queue);

    DkResult r = dkFenceWait(&fence, (int64_t)FENCE_TIMEOUT_NS);
    uint64_t t1 = armGetSystemTick();
    if (outNs) *outNs = armTicksToNs(t1 - t0);

    // FAHHHHHHH
    if (r != DkResult_Success)
        return dkQueueIsInErrorState(c->queue) ? RUN_QUEUE_ERROR : RUN_TIMEOUT;
    if (dkQueueIsInErrorState(c->queue))
        return RUN_QUEUE_ERROR;

    *out = *c->resultCpu[slot];
    return RUN_OK;
}

// Gets reports from GPU results, mainly fault records
static void reportRecords(Ctx* c, uint32_t slot, uint32_t errCount)
{
    uint32_t shown = errCount < MAX_RECORDS ? errCount : MAX_RECORDS;
    if (shown > 3) shown = 3;   

    for (uint32_t i = 0; i < shown; i++) {
        GpuRecord r = c->recCpu[slot][i];
        note("  rec thread %u %s got %08x want %08x", r.id,
             r.stage == 1u ? "ALU" : "digest", r.got, r.want);
    }
    if (errCount > shown)
        note("  (%u more records)", errCount - shown);
}


// Seperated because this way we can distinguish between source corruption and the GPU on a bender.
static uint32_t checkDataIntegrity(Ctx* c)
{
    const uint32_t stride = 997;  
    uint32_t bad = 0;
    for (uint64_t i = 0; i < N_ELEMS; i += stride)
        if (c->dataCpu[i] != dataBits(i)) bad++;
    return bad;
}


// Capture golden values to confirm them as valid.
// Runs which cannot reproduce their own digests as valid 3 times are assumed as always unstable.
// Used to refuse baking in bad data and instead re compute.
static bool calibrate(Ctx* c, Params base, uint32_t* goldDigest, uint32_t* goldMismatch)
{
    GpuResult r;
    Params p = base;

    // Capture data
    p.mode = 0;                     
    recordBatches(c, p, 1);
    if (runSync(c, 0, &r, NULL) != RUN_OK) {
        note("calibration: capture run failed");
        return false;
    }

    // Compare against what we have
    p.mode = 1;                       
    recordBatches(c, p, 1);

    uint32_t d[3], m[3];
    for (int i = 0; i < 3; i++) {
        if (runSync(c, 0, &r, NULL) != RUN_OK) {
            note("calibration: confirm run %d failed", i + 1);
            return false;
        }
        d[i] = r.digest;
        m[i] = r.mismatches;
        if (r.selfFail || r.errCount) {
            g_tally.selfFail += r.selfFail;
            note("calibration: run %d reported %u selfFail, %u records",
                 i + 1, r.selfFail, r.errCount);
            reportRecords(c, 0, r.errCount);
        }
    }

    if (d[0] == d[1] && d[1] == d[2]) {
        *goldDigest = d[0];
        *goldMismatch = m[0];
        return true;
    }

    // Weigh the results, if nothing appeared twice, then flag these values.
    g_tally.detected++;
    if (d[0] == d[1] || d[0] == d[2])      { *goldDigest = d[0]; *goldMismatch = m[0]; }
    else if (d[1] == d[2])                 { *goldDigest = d[1]; *goldMismatch = m[1]; }
    else                                   { *goldDigest = d[0]; *goldMismatch = m[0]; }
    note("CALIBRATION UNSTABLE: %08x %08x %08x -- reference may be wrong",
         d[0], d[1], d[2]);
    return true;
}

// Wait what do you mean the CPU exists
static void runDemo(PadState* pad)
{
    Ctx c;
    memset(&c, 0, sizeof(c));

    printf("CUDA95-HOS\n");
    printf("%d rows x %d threads, %llu uvec4 (%llu KiB)\n\n",
           N_ROWS, N_THREADS, (unsigned long long)N_VEC,
           (unsigned long long)(N_VEC * 16 / 1024));
    consoleUpdate(NULL);

    DkDeviceMaker devMaker;
    dkDeviceMakerDefaults(&devMaker);
    devMaker.cbDebug = debugCallback;
    DkDevice device = dkDeviceCreate(&devMaker);

    // Setup queues
    DkQueueMaker queueMaker;
    dkQueueMakerDefaults(&queueMaker, device);
    queueMaker.flags = DkQueueFlags_Compute
                     | DkQueueFlags_MediumPrio
                     | DkQueueFlags_DisableZcull;
    queueMaker.commandMemorySize = 512 * 1024;
    queueMaker.flushThreshold = 128 * 1024;
    DkQueue queue = dkQueueCreate(&queueMaker);
    c.queue = queue;

    // Allocate and create memory regions
    DkMemBlockMaker mbMaker;
    dkMemBlockMakerDefaults(&mbMaker, device, CODEMEM_SIZE);
    mbMaker.flags = DkMemBlockFlags_CpuUncached
                  | DkMemBlockFlags_GpuCached
                  | DkMemBlockFlags_Code;
    BumpAlloc codeMem = { dkMemBlockCreate(&mbMaker), 0, CODEMEM_SIZE };

    uint32_t dataSize = (uint32_t)(N_ELEMS * sizeof(uint32_t));   // 4 MiB
    uint32_t expSize = N_THREADS * sizeof(uint32_t);             // 16 KiB
    uint32_t recSize = MAX_RECORDS * sizeof(GpuRecord);          // 1 KiB
    uint32_t dataMemSize = ALIGN_UP(CMDMEM_SIZE + dataSize + expSize
                                    + NUM_INFLIGHT * (recSize + 0x100) + 0x4000,
                                    DK_MEMBLOCK_ALIGNMENT);

    dkMemBlockMakerDefaults(&mbMaker, device, dataMemSize);
    mbMaker.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
    BumpAlloc dataMem = { dkMemBlockCreate(&mbMaker), 0, dataMemSize };

    DkCmdBufMaker cbMaker;
    dkCmdBufMakerDefaults(&cbMaker, device);
    c.cmdbuf = dkCmdBufCreate(&cbMaker);

    uint32_t cmdOff = bumpAlloc(&dataMem, CMDMEM_SIZE, DK_CMDMEM_ALIGNMENT);
    dkCmdBufAddMemory(c.cmdbuf, dataMem.block, cmdOff, CMDMEM_SIZE);

    c.uboSize = ALIGN_UP(sizeof(Params), DK_UNIFORM_BUF_ALIGNMENT);
    uint32_t uboOff = bumpAlloc(&dataMem, c.uboSize, DK_UNIFORM_BUF_ALIGNMENT);
    c.uboAddr = gpuPtr(&dataMem, uboOff);

    // Allocate the proper queues so we don't stall
    for (uint32_t s = 0; s < NUM_INFLIGHT; s++) {
        uint32_t ro = bumpAlloc(&dataMem, sizeof(GpuResult), 0x100);
        c.resultAddr[s] = gpuPtr(&dataMem, ro);
        c.resultCpu[s] = (GpuResult*)cpuPtr(&dataMem, ro);

        uint32_t eo = bumpAlloc(&dataMem, recSize, 0x100);
        c.recAddr[s] = gpuPtr(&dataMem, eo);
        c.recCpu[s] = (GpuRecord*)cpuPtr(&dataMem, eo);
    }

    uint32_t expOff = bumpAlloc(&dataMem, expSize, 0x100);
    c.expAddr = gpuPtr(&dataMem, expOff);
    c.expSize = expSize;

    uint32_t dataOff = bumpAlloc(&dataMem, dataSize, 0x100);
    c.dataAddr = gpuPtr(&dataMem, dataOff);
    c.dataSize = dataSize;
    c.dataCpu = (uint32_t*)cpuPtr(&dataMem, dataOff);

    // Fill memory
    printf("filling %u KiB of test data...\n", dataSize / 1024);
    consoleUpdate(NULL);
    for (uint64_t i = 0; i < N_ELEMS; i++)
        c.dataCpu[i] = dataBits(i);

    if (!loadShader(&c.shader, &codeMem, "romfs:/shaders/stress_csh.dksh")) {
        note("could not load stress_csh.dksh, aborting");
        return;
    }

    // inital values for calibration
    int profile = 0;
    uint32_t repeats = 4;
    uint32_t goldDigest = 0, goldMismatch = 0;

    bool injectFault = false;

    // Base calibration
    Params base = { N_ROWS, repeats,
                    kProfiles[profile].alu, kProfiles[profile].gather,
                    MAX_RECORDS, 1, scatterMaskFor(profile), 0 };

    printf("calibrating (%s, repeats %u)...\n", kProfiles[profile].name, repeats);
    consoleUpdate(NULL);
    if (!calibrate(&c, base, &goldDigest, &goldMismatch)) {
        note("initial calibration failed, aborting");
        return;
    }
    printf("golden digest %08x, mismatches %u\n", goldDigest, goldMismatch);
    consoleUpdate(NULL);
    svcSleepThread(2ULL * 1000000000ULL);

    // Dispatches are idential and use atomicAdd, so batches must be DISPATCHES_PER_lIST * 1 dispatch.
    uint32_t expDigest = goldDigest * DISPATCHES_PER_LIST;
    uint32_t expMismatch = goldMismatch * DISPATCHES_PER_LIST;

    Params runParams = base;
    runParams.mode = 1;
    recordBatches(&c, runParams, DISPATCHES_PER_LIST);

    // Actually kill consoles now
    // In full seriousness I really wouldn't be surprised if some black swan event happened that did cause a console to explode.
    DkFence fences[NUM_INFLIGHT];
    memset(fences, 0, sizeof(fences));

    // Say you're alive
    // I'm alive
    // oh mah gaw
    uint64_t submitted = 0;
    uint64_t windowDisp = 0;
    uint64_t tWindow = armGetSystemTick();
    double dps = 0.0;
    bool dirty = false;
    bool fatal = false;
    const char* status  = "running";

    consoleClear();
    g_hud = true;
    drawAt(ROW_TITLE, "CUDA-95NX, GPU Stress/Stability   %d rows x %d thr",
           N_ROWS, N_THREADS);
    drawAt(ROW_LOGHDR, "RECENT EVENTS");
    drawAt(ROW_HELP, "Up/Down: Repeats   Y: Profile   X: Inject fault   Plus: Quit");
    drawTally();
    drawLog();

    while (appletMainLoop() && !fatal) {
        padUpdate(pad);
        uint64_t down = padGetButtonsDown(pad);
        if (down & HidNpadButton_Plus) break;

        if (down & HidNpadButton_Up) { if (repeats < 256) { repeats *= 2; dirty = true; } }
        if (down & HidNpadButton_Down) { if (repeats > 1)   { repeats /= 2; dirty = true; } }
        if (down & HidNpadButton_Y) { profile = (profile + 1) % NUM_PROFILES; dirty = true; }

        if (down & HidNpadButton_X) {
            // Fault injection, relies on chain_bias.
            // Only affects the redundant comps, so no recalibration required.
            // Shrimple way to realize you downed 5 grams of shrooms this morning.
            injectFault = !injectFault;
            base.chain_bias = injectFault ? 0xA5A5A5A5u : 0u;
            dkQueueWaitIdle(queue);
            runParams = base;
            runParams.mode = 1;
            recordBatches(&c, runParams, DISPATCHES_PER_LIST);
            memset(fences, 0, sizeof(fences));
            submitted = 0;
            note("fault injection %s", injectFault ? "ON" : "off");
            drawLog();
        }

        if (dirty) {
            // If we get an error, recapture golden digest for comparisons to actually work.
            dkQueueWaitIdle(queue);

            base.n_repeats = repeats;
            base.alu_iters = kProfiles[profile].alu;
            base.gather_iters = kProfiles[profile].gather;
            base.scatter_mask = scatterMaskFor(profile);
            base.mode = 1;

            // I got this
            drawAt(ROW_STATUS, "recalibrating (%s, repeats %u)...",
                   kProfiles[profile].name, repeats);
            consoleUpdate(NULL);

            // FAHHHHHHHHHHHHH
            if (!calibrate(&c, base, &goldDigest, &goldMismatch)) {
                note("recalibration failed");
                status = "STOPPED - recalibration failed";
                fatal = true;
                break;
            }
            expDigest = goldDigest* DISPATCHES_PER_LIST;
            expMismatch = goldMismatch * DISPATCHES_PER_LIST;
            note("calibrated %s repeats %u -> digest %08x",
                 kProfiles[profile].name, repeats, goldDigest);

            runParams = base;
            runParams.mode = 1;
            recordBatches(&c, runParams, DISPATCHES_PER_LIST);

            memset(fences, 0, sizeof(fences));
            submitted = 0;
            dirty = false;
            drawAt(ROW_STATUS, " ");
            drawTally();
            drawLog();
        }

        uint32_t slot = (uint32_t)(submitted % NUM_INFLIGHT);

        if (submitted >= NUM_INFLIGHT) {
            DkResult r = dkFenceWait(&fences[slot], (int64_t)FENCE_TIMEOUT_NS);

            // I have to ask, how in the holy mother of FUCK do you even see this error
            // I mean genuinely, because I have not been able to get this apart from deliberatly glitching ICs, and even then, it fucking crashes before the GPU can display a damn thing.
            if (dkQueueIsInErrorState(queue)) {
                note("FATAL: queue entered error state");
                status = "STOPPED - GPU queue error, how did we get here?";
                fatal = true;
                break;
            }
            if (r != DkResult_Success) {
                g_tally.timeouts++;
                note("FATAL: batch exceeded %llu s deadline",
                     (unsigned long long)(FENCE_TIMEOUT_NS / 1000000000ULL));
                status = "STOPPED - batch timeout";
                fatal = true;
                break;
            }

            g_tally.batches++;
            GpuResult res = *c.resultCpu[slot];

            bool bad = (res.digest != expDigest) || (res.mismatches != expMismatch);

            // Shader COMP/DUP still occurs even if digests are bad.
            // Allows to distinguish what died.
            if (res.selfFail) {
                g_tally.selfFail += res.selfFail;
                note("batch %llu: %u ALU self-check failures",
                     (unsigned long long)g_tally.batches, res.selfFail);
                bad = true;
            }
            if (res.errCount) {
                reportRecords(&c, slot, res.errCount);
                bad = true;
            }

            // We outa stability
            if (bad) {
                if (res.digest != expDigest)
                    note("batch %llu: digest %08x want %08x",
                         (unsigned long long)g_tally.batches, res.digest, expDigest);

                g_tally.detected++;

                // Re run batches twice and average, mainly for transience.
                dkQueueWaitIdle(queue);
                GpuResult a, b;
                RunStatus s1 = runSync(&c, slot, &a, NULL);
                RunStatus s2 = runSync(&c, slot, &b, NULL);
                if (s1 != RUN_OK || s2 != RUN_OK) {
                    note("FATAL: re-run failed during vote");
                    status = "STOPPED - re-run failed";
                    fatal = true;
                    break;
                }

                // If we rerun, replay the same list so batches are compared to what they should result in, not the golden value for single dispatches
                bool aOk = (a.digest == expDigest) && !a.selfFail && !a.errCount;
                bool bOk = (b.digest == expDigest) && !b.selfFail && !b.errCount;

                // If we detect transience, then we sorta ball, if its persistent, fucking overvolt.
                if (aOk && bOk) {
                    g_tally.transient++;
                    note("  -> transient, corrected");
                } else {
                    g_tally.persistent++;
                    note("  -> PERSISTENT (re-runs %08x, %08x)", a.digest, b.digest);
                }

                // How did this happen
                // A long time ago
                uint32_t badElems = checkDataIntegrity(&c);
                if (badElems) {
                    g_tally.dataBad += badElems;
                    note("  -> input buffer: %u sampled elements wrong", badElems);
                }

                drawTally();
                drawLog();
                memset(fences, 0, sizeof(fences));
                submitted = 0;
                continue;
            }

            // que
            if ((g_tally.batches % DATA_CHECK_INTERVAL) == 0) {
                uint32_t badElems = checkDataIntegrity(&c);
                if (badElems) {
                    g_tally.dataBad += badElems;
                    note("input buffer: %u sampled elements wrong", badElems);
                    drawTally();
                    drawLog();
                }
            }
        }

        memset(c.resultCpu[slot], 0, sizeof(GpuResult));
        dkQueueSubmitCommands(queue, c.batch[slot]);
        dkQueueSignalFence(queue, &fences[slot], true);
        dkQueueFlush(queue);

        submitted++;
        windowDisp += DISPATCHES_PER_LIST;

        uint64_t now = armGetSystemTick();
        uint64_t windowNs = armTicksToNs(now - tWindow);
        if (windowNs >= 500000000ULL) {
            double secs = windowNs / 1.0e9;
            dps = windowDisp / secs;

            // Dispaly profiles/text, also stats.
            double bytesPer = (double)N_THREADS * 16.0 * repeats
                            * ((double)N_ROWS + (double)kProfiles[profile].gather);

            drawAt(ROW_STATS + 0, "profile   : %-8s repeats %-4u alu %-4u gather %-4u win %uKiB%s",
                   kProfiles[profile].name, repeats,
                   kProfiles[profile].alu, kProfiles[profile].gather,
                   (unsigned)((scatterMaskFor(profile) + 1u) * 16u / 1024u),
                   injectFault ? "  [INJECT]" : "");
            drawAt(ROW_STATS + 1, "batches   : %-12llu  %.1f disp/s",
                   (unsigned long long)g_tally.batches, dps);
            drawAt(ROW_STATS + 2, "read rate : %.2f GB/s   golden %08x",
                   (dps * bytesPer) / 1.0e9, goldDigest);
            drawAt(ROW_STATS + 3, "elapsed   : %llu s",
                   (unsigned long long)elapsedSec());
            drawTally();

            windowDisp = 0;
            tWindow = now;
        }

        consoleUpdate(NULL);
    }

    if (!fatal) status = "stopped by user";

    drawTally();
    drawLog();
    drawAt(ROW_STATUS, "%s after %llu batches, %llu s",
           status, (unsigned long long)g_tally.batches,
           (unsigned long long)elapsedSec());
    consoleUpdate(NULL);

    if (!dkQueueIsInErrorState(queue))
        dkQueueWaitIdle(queue);
    dkCmdBufDestroy(c.cmdbuf);
    dkMemBlockDestroy(dataMem.block);
    dkMemBlockDestroy(codeMem.block);
    dkQueueDestroy(queue);
    dkDeviceDestroy(device);
}

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    g_tStart = armGetSystemTick();

    consoleInit(NULL);
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);

    Result rc = romfsInit();
    if (R_FAILED(rc)) {
        printf("romfsInit failed: 0x%x\n", rc);
    } else {
        runDemo(&pad);
        romfsExit();
    }

    if (g_hud)
        drawAt(ROW_STATUS + 2, "Press + to exit.");
    else
        printf("\nPress + to exit.\n");
    consoleUpdate(NULL);

    while (appletMainLoop()) {
        padUpdate(&pad);
        if (padGetButtonsDown(&pad) & HidNpadButton_Plus) break;
        consoleUpdate(NULL);
    }

    consoleExit(NULL);
    return 0;
}

// Special thanks to Ray.