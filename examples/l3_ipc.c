#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define L3_IPC_SHM_NAME "/vcache_l3_ipc"
#define L3_IPC_MAGIC 0x4C33495043303031ULL
#define L3_IPC_TAG_SEED 0x732641U
#define L3_IPC_SLOTS 2048U
#define L3_IPC_MASK (L3_IPC_SLOTS - 1U)
#define L3_IPC_PAYLOAD_MAX 488U

typedef struct __attribute__((aligned(128))) {
    atomic_uint_fast64_t seq;
    uint64_t ts_ns;
    uint32_t payload_len;
    uint32_t checksum;
    char payload[L3_IPC_PAYLOAD_MAX];
} l3_ipc_slot_t;

typedef struct {
    alignas(128) uint64_t magic;
    uint32_t tag_seed;
    uint32_t num_slots;
    alignas(128) atomic_uint_fast64_t write_seq;
    atomic_uint_fast64_t total_bytes;
    atomic_uint_fast64_t last_publish_ns;
    alignas(128) l3_ipc_slot_t slots[L3_IPC_SLOTS];
} l3_ipc_ring_t;

static l3_ipc_ring_t* g_ring = NULL;

static inline uint64_t mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint32_t compute_tag(const char* buf, uint32_t len) {
    uint32_t h = L3_IPC_TAG_SEED;
    for (uint32_t i = 0; i < len; i++) {
        h = (h * 16777619U) ^ (uint8_t)buf[i];
    }
    return h;
}

static l3_ipc_ring_t* ensure_ring_mapped(void) {
    if (g_ring != NULL) {
        return g_ring;
    }
    int fd = shm_open(L3_IPC_SHM_NAME, O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) == 0 && (size_t)st.st_size < sizeof(l3_ipc_ring_t)) {
        if (ftruncate(fd, sizeof(l3_ipc_ring_t)) != 0) {
            close(fd);
            return NULL;
        }
    }
    void* ptr = mmap(NULL, sizeof(l3_ipc_ring_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        return NULL;
    }
    l3_ipc_ring_t* ring = (l3_ipc_ring_t*)ptr;
    if (ring->magic != L3_IPC_MAGIC) {
        ring->tag_seed = L3_IPC_TAG_SEED;
        ring->num_slots = L3_IPC_SLOTS;
        atomic_store_explicit(&ring->write_seq, 0, memory_order_relaxed);
        atomic_store_explicit(&ring->total_bytes, 0, memory_order_relaxed);
        atomic_store_explicit(&ring->last_publish_ns, 0, memory_order_relaxed);
        ring->magic = L3_IPC_MAGIC;
        atomic_thread_fence(memory_order_release);
    }
    g_ring = ring;
    return g_ring;
}

int l3_ipc_publish(const char* payload, uint32_t len, uint64_t* elapsed_ns_out) {
    if (!payload) return -1;
    l3_ipc_ring_t* ring = ensure_ring_mapped();
    if (!ring) return -2;

    uint64_t t0 = mono_ns();
    if (len > L3_IPC_PAYLOAD_MAX - 1) {
        len = L3_IPC_PAYLOAD_MAX - 1;
    }

    uint64_t ticket = atomic_fetch_add_explicit(&ring->write_seq, 1, memory_order_acq_rel);
    l3_ipc_slot_t* slot = &ring->slots[ticket & L3_IPC_MASK];

    atomic_store_explicit(&slot->seq, ticket * 2ULL + 1ULL, memory_order_release);
    slot->ts_ns = t0;
    slot->payload_len = len;
    slot->checksum = compute_tag(payload, len);
    memcpy(slot->payload, payload, len);
    slot->payload[len] = '\0';

    atomic_store_explicit(&slot->seq, ticket * 2ULL + 2ULL, memory_order_release);

    uint64_t dt = mono_ns() - t0;
    atomic_fetch_add_explicit(&ring->total_bytes, len, memory_order_relaxed);
    atomic_store_explicit(&ring->last_publish_ns, dt, memory_order_relaxed);

    if (elapsed_ns_out) {
        *elapsed_ns_out = dt;
    }
    return 0;
}

uint64_t l3_ipc_get_write_seq(void) {
    l3_ipc_ring_t* ring = ensure_ring_mapped();
    if (!ring) return 0;
    return atomic_load_explicit(&ring->write_seq, memory_order_acquire);
}

int l3_ipc_read_latest(int max_items, char* out_buf, int out_buf_cap) {
    if (!out_buf || out_buf_cap <= 1 || max_items <= 0) return 0;
    l3_ipc_ring_t* ring = ensure_ring_mapped();
    if (!ring) return 0;

    uint64_t w_seq = atomic_load_explicit(&ring->write_seq, memory_order_acquire);
    if (w_seq == 0) {
        out_buf[0] = '\0';
        return 0;
    }

    uint64_t count = (w_seq < L3_IPC_SLOTS) ? w_seq : L3_IPC_SLOTS;
    if ((uint64_t)max_items < count) {
        count = (uint64_t)max_items;
    }
    uint64_t start_ticket = w_seq - count;

    int written = 0;
    int items = 0;
    for (uint64_t t = start_ticket; t < w_seq; t++) {
        l3_ipc_slot_t* slot = &ring->slots[t & L3_IPC_MASK];
        uint64_t s1 = atomic_load_explicit(&slot->seq, memory_order_acquire);
        if (s1 != t * 2ULL + 2ULL) {
            continue;
        }
        uint32_t plen = slot->payload_len;
        if (plen == 0 || plen >= L3_IPC_PAYLOAD_MAX) {
            continue;
        }
        if (written + (int)plen + 2 >= out_buf_cap) {
            break;
        }
        memcpy(out_buf + written, slot->payload, plen);
        uint64_t s2 = atomic_load_explicit(&slot->seq, memory_order_acquire);
        if (s1 != s2) {
            continue;
        }
        if (compute_tag(out_buf + written, plen) != slot->checksum) {
            continue;
        }
        written += (int)plen;
        out_buf[written++] = '\n';
        items++;
    }
    out_buf[written] = '\0';
    return items;
}
