/**
 * Accounts: sign in, register, keep the token.
 *
 * The token goes into the first WebSocket message, never a URL (CLAUDE.md 7),
 * and into an Authorization header for REST. It is kept in localStorage so a
 * reload does not sign the learner out; it expires on the server's schedule
 * (JWT_TTL_MIN), after which the socket answers UNAUTHENTICATED and the page
 * asks to sign in again.
 */

const STORAGE_KEY = "sign-assistant.session";

export interface Session {
  readonly token: string;
  readonly userId: number;
  readonly email: string;
}

export class AuthError extends Error {}

export function savedSession(): Session | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    // Private windows and blocked storage throw; signing in again is fine.
    return null;
  }
}

export function signOut(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing to clear.
  }
}

export function signIn(email: string, password: string): Promise<Session> {
  return authenticate("/auth/login", email, password);
}

export function register(email: string, password: string): Promise<Session> {
  return authenticate("/auth/register", email, password);
}

async function authenticate(path: string, email: string, password: string): Promise<Session> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
  } catch {
    throw new AuthError("Could not reach the server.");
  }

  if (!response.ok) throw new AuthError(await explain(response));

  const body = (await response.json()) as { token: string; user_id: number };
  const session: Session = { token: body.token, userId: body.user_id, email };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  } catch {
    // Works for this page load; the learner signs in again after a reload.
  }
  return session;
}

/** One readable sentence from the backend's error shape (CLAUDE.md 10). */
async function explain(response: Response): Promise<string> {
  const body = (await response.json().catch(() => null)) as {
    error?: { code?: string; detail?: string; field?: string };
  } | null;
  const error = body?.error;

  switch (error?.code) {
    case "BAD_CREDENTIALS":
      return "Email or password is wrong.";
    case "EMAIL_TAKEN":
      return "An account with this email already exists. Sign in instead.";
    case "WEAK_PASSWORD":
      return error.detail ?? "Choose a stronger password.";
    case "INVALID_REQUEST":
      if (error.field === "email") return "That does not look like an email address.";
      if (error.field === "password") return "The password must be 8 to 72 characters.";
      return "Please check the form.";
    default:
      return `Something went wrong (HTTP ${response.status}).`;
  }
}
