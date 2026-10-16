import "./style.css";
import { AuthError, register, savedSession, signIn, signOut, type Session } from "./auth";
import { LearnPage } from "./pages/learn";

const authForm = byId<HTMLFormElement>("auth");
const app = byId<HTMLElement>("app");
const account = byId<HTMLElement>("account");

function showApp(session: Session): void {
  authForm.hidden = true;
  app.hidden = false;
  account.hidden = false;
  byId<HTMLElement>("account-email").textContent = session.email;
  void new LearnPage(session.token, () => showSignIn("Your session ended. Please sign in again.")).init();
}

function showSignIn(reason = ""): void {
  signOut();
  // A fresh page is the simplest way to drop the camera, socket and render
  // loop of the session that just ended.
  if (!app.hidden) {
    location.reload();
    return;
  }
  app.hidden = true;
  account.hidden = true;
  authForm.hidden = false;
  byId<HTMLElement>("auth-error").textContent = reason;
  byId<HTMLElement>("status").textContent = "";
}

async function submit(action: typeof signIn): Promise<void> {
  const email = byId<HTMLInputElement>("auth-email").value.trim();
  const password = byId<HTMLInputElement>("auth-password").value;
  const error = byId<HTMLElement>("auth-error");
  error.textContent = "";
  try {
    showApp(await action(email, password));
  } catch (cause) {
    error.textContent = cause instanceof AuthError ? cause.message : String(cause);
  }
}

authForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void submit(signIn);
});
byId<HTMLButtonElement>("register").addEventListener("click", () => {
  if (authForm.reportValidity()) void submit(register);
});
byId<HTMLButtonElement>("sign-out").addEventListener("click", () => showSignIn());

const session = savedSession();
if (session) showApp(session);
else showSignIn();

function byId<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`markup is missing #${id}`);
  return found as T;
}
