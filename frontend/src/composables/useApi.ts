/**
 * Composable for typed HTTP API calls.
 *
 * Provides get/post/put/del/upload helpers that hit the /api prefix
 * and throw on non-OK responses. The Vite proxy forwards these to
 * the FastAPI backend in development.
 */

const BASE_URL = '/api'

/**
 * @returns Typed API client methods
 */
export function useApi() {
  /**
   * GET request returning typed JSON.
   * @param path - API path after /api (e.g. "/feeds")
   */
  async function get<T>(path: string): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  /**
   * POST request with optional JSON body.
   * @param path - API path after /api
   * @param data - Optional request body (will be JSON-serialized)
   */
  async function post<T>(path: string, data?: unknown): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: data ? JSON.stringify(data) : undefined,
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  /**
   * PUT request with JSON body.
   * @param path - API path after /api
   * @param data - Request body (will be JSON-serialized)
   */
  async function put<T>(path: string, data: unknown): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  /**
   * DELETE request.
   * Returns null for 204 No Content responses.
   * @param path - API path after /api
   */
  async function del<T = void>(path: string): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(await res.text())
    if (res.status === 204) return undefined as T
    return res.json()
  }

  /**
   * Upload a file via multipart form data.
   * @param path - API path after /api
   * @param file - File object to upload
   * @param fieldName - Form field name (default: "file")
   */
  async function upload<T>(path: string, file: File, fieldName = 'file'): Promise<T> {
    const formData = new FormData()
    formData.append(fieldName, file)
    const res = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  return { get, post, put, del, upload }
}
