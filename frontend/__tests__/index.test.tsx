import { render, screen } from '@testing-library/react'
import Home from '../pages/index'

test('renders title and buttons', () => {
  render(<Home />)
  expect(screen.getByText(/Industrial Monitor Pro/i)).toBeInTheDocument()
  expect(screen.getByText(/Calibrate Dial/i)).toBeInTheDocument()
  expect(screen.getByText(/Export CSV/i)).toBeInTheDocument()
})