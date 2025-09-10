import { render, screen } from '@testing-library/react'
import CameraTile from '../components/CameraTile'

test('shows camera title, status badge and thresholds button', () => {
  render(<CameraTile id="camX" title="Camera X" onRemove={() => {}} />)
  expect(screen.getByText(/Camera X/)).toBeInTheDocument()
  expect(screen.getByText(/UNKNOWN|NORMAL|LOW|MEDIUM|HIGH/)).toBeTruthy()
  expect(screen.getByText(/Thresholds/)).toBeInTheDocument()
})