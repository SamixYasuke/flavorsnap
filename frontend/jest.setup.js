import '@testing-library/jest-dom'

// Mock IntersectionObserver
class IntersectionObserver {
  observe() { return null }
  unobserve() { return null }
  disconnect() { return null }
}
window.IntersectionObserver = IntersectionObserver