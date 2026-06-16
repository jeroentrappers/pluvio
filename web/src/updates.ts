// Live "new prediction" notifications over WebSocket. The server pushes the
// published snapshot id whenever the worker advances it (~every 5 min), so the
// client refetches on demand instead of polling. Reconnects with backoff; if the
// socket can't be established the app still works (manual refresh / re-locate).
import { API_BASE } from './config'

const WS_URL = `${API_BASE.replace(/^http/, 'ws')}/v1/ws`

interface SnapshotMessage {
  type: 'snapshot'
  snapshot: string
  issued_at?: string | null
}

// Subscribe to new-snapshot notices. `onSnapshot` fires with the snapshot id on
// every message (the caller dedupes against what it has loaded). Returns an
// unsubscribe function that closes the socket and stops reconnecting.
export function subscribeUpdates(onSnapshot: (snapshot: string) => void): () => void {
  let ws: WebSocket | null = null
  let retry = 0
  let timer: ReturnType<typeof setTimeout> | null = null
  let closed = false

  const connect = () => {
    if (closed) return
    try {
      ws = new WebSocket(WS_URL)
    } catch {
      scheduleReconnect()
      return
    }
    ws.onopen = () => {
      retry = 0
    }
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as SnapshotMessage
        if (msg?.type === 'snapshot' && msg.snapshot) onSnapshot(msg.snapshot)
      } catch {
        /* ignore malformed frames */
      }
    }
    ws.onclose = () => scheduleReconnect()
    ws.onerror = () => ws?.close()
  }

  const scheduleReconnect = () => {
    if (closed || timer) return
    // Exponential backoff, capped at 30 s, so a flapping server isn't hammered.
    const delay = Math.min(30_000, 1000 * 2 ** retry)
    retry += 1
    timer = setTimeout(() => {
      timer = null
      connect()
    }, delay)
  }

  connect()

  return () => {
    closed = true
    if (timer) clearTimeout(timer)
    ws?.close()
  }
}
